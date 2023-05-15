// Copyright (c) 2021 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <sys/time.h>
// #include <time.h>
#include <chrono>


#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb/stb_image_resize.h>

#include "rknn_api.h"
#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>

#include "rga.h"
#include "drm_func.h"
#include "rga_func.h"
#include "rknn_api.h"

using namespace std;

int num_class = 1;
int img_size = 640;
float objThreshold = 0.5;
float class_score = 0.5;
float nmsThreshold = 0.5;
std::vector<string> class_names = {"mask"};
std::vector<int> stride = {8,16,32};
std::vector<std::vector<int>> anchors = {{10,13,16,30,33,23},{30,61,62,45,59,119},{116,90,156,198,373,326}};

typedef struct BoxInfo
{
	float x1;
	float y1;
	float x2;
	float y2;
	float score;
	int label;
} BoxInfo;

void nms(vector<BoxInfo>& input_boxes)
{
	sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
	vector<float> vArea(input_boxes.size());
	for (int i = 0; i < int(input_boxes.size()); ++i)
	{
		vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
			* (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
	}

	vector<bool> isSuppressed(input_boxes.size(), false);
	for (int i = 0; i < int(input_boxes.size()); ++i)
	{
		if (isSuppressed[i]) { continue; }
		for (int j = i + 1; j < int(input_boxes.size()); ++j)
		{
			if (isSuppressed[j]) { continue; }
			float xx1 = (max)(input_boxes[i].x1, input_boxes[j].x1);
			float yy1 = (max)(input_boxes[i].y1, input_boxes[j].y1);
			float xx2 = (min)(input_boxes[i].x2, input_boxes[j].x2);
			float yy2 = (min)(input_boxes[i].y2, input_boxes[j].y2);

			float w = (max)(float(0), xx2 - xx1 + 1);
			float h = (max)(float(0), yy2 - yy1 + 1);
			float inter = w * h;
			float ovr = inter / (vArea[i] + vArea[j] - inter);

			if (ovr >= nmsThreshold)
			{
				isSuppressed[j] = true;
			}
		}
	}
	// return post_nms;
	int idx_t = 0;
	input_boxes.erase(remove_if(input_boxes.begin(), input_boxes.end(), [&idx_t, &isSuppressed](const BoxInfo& f) { return isSuppressed[idx_t++]; }), input_boxes.end());
}

/*-------------------------------------------
                  Functions
-------------------------------------------*/

static void printRKNNTensor(rknn_tensor_attr *attr)
{
    printf("index=%d name=%s n_dims=%d dims=[%d %d %d %d] n_elems=%d size=%d fmt=%d type=%d qnt_type=%d fl=%d zp=%d scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[3], attr->dims[2], attr->dims[1], attr->dims[0],
           attr->n_elems, attr->size, 0, attr->type, attr->qnt_type, attr->fl, attr->zp, attr->scale);
}

static unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp = fopen(filename, "rb");
    if (fp == nullptr)
    {
        printf("fopen %s fail!\n", filename);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    int model_len = ftell(fp);
    unsigned char *model = (unsigned char *)malloc(model_len);
    fseek(fp, 0, SEEK_SET);
    if (model_len != fread(model, 1, model_len, fp))
    {
        printf("fread %s fail!\n", filename);
        free(model);
        return NULL;
    }
    *model_size = model_len;
    if (fp)
    {
        fclose(fp);
    }
    return model;
}

static int rknn_GetTop(
    float *pfProb,
    float *pfMaxProb,
    uint32_t *pMaxClass,
    uint32_t outputCount,
    uint32_t topNum)
{
    uint32_t i, j;

#define MAX_TOP_NUM 20
    if (topNum > MAX_TOP_NUM)
        return 0;

    memset(pfMaxProb, 0, sizeof(float) * topNum);
    memset(pMaxClass, 0xff, sizeof(float) * topNum);

    for (j = 0; j < topNum; j++)
    {
        for (i = 0; i < outputCount; i++)
        {
            if ((i == *(pMaxClass + 0)) || (i == *(pMaxClass + 1)) || (i == *(pMaxClass + 2)) ||
                (i == *(pMaxClass + 3)) || (i == *(pMaxClass + 4)))
            {
                continue;
            }

            if (pfProb[i] > *(pfMaxProb + j))
            {
                *(pfMaxProb + j) = pfProb[i];
                *(pMaxClass + j) = i;
            }
        }
    }

    return 1;
}

static unsigned char *load_image(const char *image_path, rknn_tensor_attr *input_attr)
{
    int req_height = 0;
    int req_width = 0;
    int req_channel = 0;

    switch (input_attr->fmt)
    {
    case RKNN_TENSOR_NHWC:
        req_height = input_attr->dims[2];
        req_width = input_attr->dims[1];
        req_channel = input_attr->dims[0];
        break;
    case RKNN_TENSOR_NCHW:
        req_height = input_attr->dims[1];
        req_width = input_attr->dims[0];
        req_channel = input_attr->dims[2];
        break;
    default:
        printf("meet unsupported layout\n");
        return NULL;
    }

    printf("w=%d,h=%d,c=%d, fmt=%d\n", req_width, req_height, req_channel, input_attr->fmt);

    int height = 0;
    int width = 0;
    int channel = 0;

    unsigned char *image_data = stbi_load(image_path, &width, &height, &channel, req_channel);
    if (image_data == NULL)
    {
        printf("load image failed!\n");
        return NULL;
    }

    if (width != req_width || height != req_height)
    {
        unsigned char *image_resized = (unsigned char *)STBI_MALLOC(req_width * req_height * req_channel);
        if (!image_resized)
        {
            printf("malloc image failed!\n");
            STBI_FREE(image_data);
            return NULL;
        }
        if (stbir_resize_uint8(image_data, width, height, 0, image_resized, req_width, req_height, 0, channel) != 1)
        {
            printf("resize image failed!\n");
            STBI_FREE(image_data);
            return NULL;
        }
        STBI_FREE(image_data);
        image_data = image_resized;
    }

    return image_data;
}

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv)
{
    rknn_context ctx;
    int ret;
    int model_len = 0;
    unsigned char *model;

    const char *model_path = argv[1];
    const char *img_path = argv[2];

    // Load RKNN Model
    model = load_model(model_path, &model_len);

    ret = rknn_init(&ctx, model, model_len, 0);
    if (ret < 0)
    {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    // Get Model Input Output Info


    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    printf("input tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        printRKNNTensor(&(input_attrs[i]));
    }

    printf("output tensors:\n");

    
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        printRKNNTensor(&(output_attrs[i]));
        printf("%d, %d, %d, %d\n",output_attrs[i].dims[0],output_attrs[i].dims[1],output_attrs[i].dims[2],output_attrs[i].dims[3]);
    }
    vector<BoxInfo> generate_boxes;
    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    unsigned char *input_data = NULL;

    rga_context rga_ctx;
    drm_context drm_ctx;
    memset(&rga_ctx, 0, sizeof(rga_context));
    memset(&drm_ctx, 0, sizeof(drm_context));

    // DRM alloc buffer
    int drm_fd = -1;
    int buf_fd = -1; // converted from buffer handle
    unsigned int handle;
    size_t actual_size = 0;
    void *drm_buf = NULL;

    cv::Mat input_img = cv::imread(img_path);
    int video_width = input_img.cols;
    int video_height = input_img.rows;
    int channel = input_img.channels();

    int width = img_size;
    int height = img_size;

    drm_fd = drm_init(&drm_ctx);
    drm_buf = drm_buf_alloc(&drm_ctx, drm_fd, video_width, video_height, channel * 8, &buf_fd, &handle, &actual_size);

    void *resize_buf = malloc(height * width * channel);
    // init rga context
       
    RGA_init(&rga_ctx);

    uint32_t input_model_image_size = width * height * channel;
    // Set Input Data
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = input_model_image_size;
    inputs[0].fmt = RKNN_TENSOR_NHWC;

    for(int run_times =0;run_times<10;run_times++)
    {
        // clock_t start,end;
        // start = clock();
        auto start=std::chrono::steady_clock::now();
        // Load image
        
        // input_data = load_image(img_path, &input_attrs[0]);
        // if (!input_data)
        // {
        //     return -1;
        // }

        cv::Mat input_img = cv::imread(img_path);
        cv::cvtColor(input_img, input_img, cv::COLOR_BGR2RGB);
        // clock_t load_img_time = clock();
        auto load_img_time=std::chrono::steady_clock::now();
        double dr_ms=std::chrono::duration<double,std::milli>(load_img_time-start).count();
        std::cout<<"load_img_time = "<< dr_ms << std::endl;
        // cout<<"load_img_time = "<<double(load_img_time-start)/CLOCKS_PER_SEC*1000<<"ms"<<endl;
/*
        // Set Input Data
        rknn_input inputs[1];
        memset(inputs, 0, sizeof(inputs));
        inputs[0].index = 0;
        inputs[0].type = RKNN_TENSOR_UINT8;
        inputs[0].size = input_attrs[0].size;
        inputs[0].fmt = RKNN_TENSOR_NHWC;
        inputs[0].buf = input_img.data;//input_data;

        ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
        if (ret < 0)
        {
            printf("rknn_input_set fail! ret=%d\n", ret);
            return -1;
        }
*/

        memcpy(drm_buf, (uint8_t *)input_img.data , video_width * video_height * channel);
       
        
        img_resize_slow(&rga_ctx, drm_buf, video_width, video_height, resize_buf, width, height);
       
        inputs[0].buf = resize_buf;
        ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
        if (ret < 0)
        {
            printf("ERROR: rknn_inputs_set fail! ret=%d\n", ret);
            return NULL;
        }
       


        // clock_t pre_time = clock();
        // cout<<"pre_time = "<<double(pre_time-load_img_time)/CLOCKS_PER_SEC*1000<<"ms"<<endl;
        auto pre_time=std::chrono::steady_clock::now();
        dr_ms=std::chrono::duration<double,std::milli>(pre_time-load_img_time).count();
        std::cout<<"pre_time = "<< dr_ms << std::endl;
        // Run
        printf("rknn_run\n");
        
        ret = rknn_run(ctx, nullptr);
        if (ret < 0)
        {
            printf("rknn_run fail! ret=%d\n", ret);
            return -1;
        }

        // clock_t rknn_run_time = clock();
        // cout<<"rknn_run_time = "<<double(rknn_run_time-pre_time)/CLOCKS_PER_SEC*1000<<"ms"<<endl;
        auto rknn_run_time=std::chrono::steady_clock::now();
        dr_ms=std::chrono::duration<double,std::milli>(rknn_run_time-pre_time).count();
        std::cout<<"rknn_run_time = "<< dr_ms << std::endl;
        
        // Get Output
        for (int i = 0; i < io_num.n_output; i++)
        {
            outputs[i].want_float = 1;
        }
        ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
        if (ret < 0)
        {
        printf("ERROR: rknn_outputs_get fail! ret=%d\n", ret);
        return NULL;
        }

        // resize ratio
        float ratioh = 1.0, ratiow = 1.0;
        int n = 0, q = 0, i = 0, j = 0, k = 0; ///xmin,ymin,xamx,ymax,box_score,class_score
        // clock_t post_time = clock();
        const int nout = num_class + 5;
        float *preds = (float *)outputs[0].buf;
        for (n = 0; n < 3; n++)   ///
        {
            int num_grid_x = (int)(img_size / stride[n]);
            int num_grid_y = (int)(img_size / stride[n]);
            for (q = 0; q < 3; q++)    ///anchor
            {
                const float anchor_w = anchors[n][q * 2];
                const float anchor_h = anchors[n][q * 2 + 1];
                for (i = 0; i < num_grid_y; i++)
                {
                    for (j = 0; j < num_grid_x; j++)
                    {
                        float box_score = preds[4];
                        if (box_score > objThreshold)
                        {
                            float class_score = 0;
                            int class_ind = 0;
                            for (k = 0; k < num_class; k++)
                            {
                                if (preds[k + 5] > class_score)
                                {
                                    class_score = preds[k + 5];
                                    class_ind = k;
                                }
                            }
                            //if (class_score > this->confThreshold)
                            //{ 
                            float cx = (preds[0] * 2.f - 0.5f + j) * stride[n];  ///cx
                            float cy = (preds[1] * 2.f - 0.5f + i) * stride[n];   ///cy
                            float w = powf(preds[2] * 2.f, 2.f) * anchor_w;   ///w
                            float h = powf(preds[3] * 2.f, 2.f) * anchor_h;  ///h

                            float xmin = (cx - 0.5 * w)*ratiow;
                            float ymin = (cy - 0.5 * h)*ratioh;
                            float xmax = (cx + 0.5 * w)*ratiow;
                            float ymax = (cy + 0.5 * h)*ratioh;
                            //printf("%f,%f,%f,%f,%f,%d\n",xmin, ymin, xmax, ymax, class_score, class_ind);
                            generate_boxes.push_back(BoxInfo{ xmin, ymin, xmax, ymax, class_score, class_ind });
                            //}
                        }
                        preds += nout;
                    }
                }
            }
        }

        nms(generate_boxes);
        // end = clock();
        // cout<<"time = "<<double(end-start)/CLOCKS_PER_SEC*1000<<"ms"<<endl;
        // cout<<"post_time = "<<double(end-post_time)/CLOCKS_PER_SEC*1000<<"ms"<<endl;
        auto end_time = std::chrono::steady_clock::now();
        dr_ms=std::chrono::duration<double,std::milli>(end_time-start).count();
        std::cout<<"end_time = "<< dr_ms << std::endl;
	}
	cv::Mat frame = cv::imread(img_path);
    cv::resize(frame,frame,cv::Size(img_size,img_size));
	for (size_t i = 0; i < generate_boxes.size(); ++i)
	{
		int xmin = int(generate_boxes[i].x1);
		int ymin = int(generate_boxes[i].y1);
		cv::rectangle(frame, cv::Point(xmin, ymin), cv::Point(int(generate_boxes[i].x2), int(generate_boxes[i].y2)), cv::Scalar(0, 0, 255), 2);
		std::string str_num = std::to_string(generate_boxes[i].score);
    	std::string label = str_num.substr(0, str_num.find(".") + 3);
		label = class_names[generate_boxes[i].label] + ":" + label;
		cv::putText(frame, label, cv::Point(xmin, ymin - 5), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0), 1);
		
	}
	cv::imwrite("output.jpg",frame);
    // Release rknn_outputs
    rknn_outputs_release(ctx, 1, outputs);

    free(resize_buf);
    drm_buf_destroy(&drm_ctx, drm_fd, buf_fd, handle, drm_buf, actual_size);
    drm_deinit(&drm_ctx, drm_fd);
    RGA_deinit(&rga_ctx);

    // Release
    if (ctx >= 0)
    {
        rknn_destroy(ctx);
    }
    if (model)
    {
        free(model);
    }

    if (input_data)
    {
        stbi_image_free(input_data);
    }

    return 0;
}
