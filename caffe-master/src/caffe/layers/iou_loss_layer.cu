/******************************************************

	File Name       : iou_loss_layer.cu
	File Description: GPU implement
	Author          : yanghongyu
	Create Time     : 2019-9-7 10:01:23

*******************************************************/
#include <vector>
	
#include "caffe/layers/iou_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
	
	namespace caffe {
	
	
	template <typename Dtype>
	__global__ void IOULossForwardGPU(const int nthreads,	
	const Dtype* d1, const Dtype* d2,	const Dtype* d3, const Dtype* d4,
	const Dtype* d1_t, const Dtype* d2_t,	const Dtype* d3_t, const Dtype* d4_t,
	const Dtype* score_map, Dtype* loss) {
		CUDA_KERNEL_LOOP(i, nthreads) {
			const int score = static_cast<int>(score_map[i]);
			
			if (score == 0) {
				loss[i] = 0;
			} else {
				Dtype w1 = (d2[i] > d2_t[i])?d2_t[i]:d2[i];
				Dtype w2 = (d4[i] > d4_t[i])?d4_t[i]:d4[i];
				Dtype intersection_w = w1 + w2;
				
				Dtype h1 = (d1[i] > d1_t[i])?d1_t[i]:d1[i];
				Dtype h2 = (d3[i] > d3_t[i])?d3_t[i]:d3[i];
				Dtype intersection_h = h1 + h2;

				Dtype intersection_area = intersection_w * intersection_h;
				

				Dtype gt_w = d2_t[i] + d4_t[i];
				Dtype gt_h = d1_t[i] + d3_t[i];

				Dtype gt_area = gt_w * gt_h;

				Dtype pred_w = d2[i] + d4[i];
				Dtype pred_h = d1[i] + d3[i];

				Dtype pred_area = pred_w * pred_h;

				if (intersection_area < 1e-8){
					intersection_area = 1.;
				}
				Dtype iou = intersection_area/(gt_area + pred_area - intersection_area);

				Dtype w_r = 1.;
				loss[i] -= w_r * log(iou);
				//printf("thread %d :  d1 d2 d3 d4 : %f, %f, %f, %f ",i, d1[i], d2[i], d3[i], d4[i]);
				//printf("w1 : %f ", w1);
				//printf("w2 : %f ", w2);
				//printf("iou : %f", iou);
				//printf("loss i : %f", loss[i]);
			}
		}
	}
	
	
	template <typename Dtype>
	__global__ void IOULossBackwardGPU(const int nthreads,	
	const Dtype* d1, const Dtype* d2,	const Dtype* d3, const Dtype* d4,
	const Dtype* d1_t, const Dtype* d2_t,	const Dtype* d3_t, const Dtype* d4_t,
	Dtype* diff1,	Dtype* diff2,	Dtype* diff3,	Dtype* diff4,
	const Dtype* score_map) {
		CUDA_KERNEL_LOOP(i, nthreads) {
			
			const int score = static_cast<int>(score_map[i]);
			if (score == 0) {
				diff1[i] = 0;
				diff2[i] = 0;
				diff3[i] = 0;
				diff4[i] = 0;
			} else {
				Dtype w1 = (d2[i] > d2_t[i])?d2_t[i]:d2[i];
				Dtype w2 = (d4[i] > d4_t[i])?d4_t[i]:d4[i];
				Dtype intersection_w = w1 + w2;
				
				Dtype h1 = (d1[i] > d1_t[i])?d1_t[i]:d1[i];
				Dtype h2 = (d3[i] > d3_t[i])?d3_t[i]:d3[i];
				Dtype intersection_h = h1 + h2;

				Dtype intersection_area = intersection_w * intersection_h;
				

				Dtype gt_w = d2_t[i] + d4_t[i];
				Dtype gt_h = d1_t[i] + d3_t[i];

				Dtype gt_area = gt_w * gt_h;

				Dtype pred_w = d2[i] + d4[i];
				Dtype pred_h = d1[i] + d3[i];

				Dtype pred_area = pred_w * pred_h;
				Dtype u_area = gt_area + pred_area - intersection_area;

				if (intersection_h < 1e-8){
					intersection_h = 1.;
				}

				if (intersection_w < 1e-8){
					intersection_w = 1.;
				}

				//Dtype w_r = 1 + 0.0001 * gt_area;
				//Dtype h_r = 1 + 0.0001 * gt_area;

				Dtype w_r = 1.;
				Dtype h_r = 1.;
				
				if(d1[i] <= d1_t[i])
					diff1[i] = h_r * ((pred_w - intersection_w)/u_area - 1./intersection_h);
				else
					diff1[i] = h_r * (pred_w/u_area);

				if(d2[i] <= d2_t[i])
					diff2[i] = w_r * ((pred_h - intersection_h)/u_area - 1./intersection_w);
				else
					diff2[i] = w_r * (pred_h/u_area);

				if(d3[i] <= d3_t[i])
					diff3[i] = h_r * ((pred_w - intersection_w)/u_area - 1./intersection_h);
				else
					diff3[i] = h_r * (pred_w/u_area);

				if(d4[i] <= d4_t[i])
					diff4[i] = w_r * ((pred_h - intersection_h)/u_area - 1./intersection_w);
				else
					diff4[i] = w_r * (pred_h/u_area);
			}
		}
	}
	
	
	
	template <typename Dtype>
	void IOULossLayer<Dtype>::Forward_gpu(
			const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

		const int count = bottom[0]->count(2);
		const Dtype* input_data = bottom[0]->gpu_data();
		const Dtype* target = bottom[1]->gpu_data();
		const Dtype* score_map = bottom[2]->gpu_data();

		const Dtype* d1 = input_data;
		const Dtype* d2 = input_data + bottom[0]->offset(0, 1, 0, 0);
		const Dtype* d3 = input_data + bottom[0]->offset(0, 2, 0, 0);
		const Dtype* d4 = input_data + bottom[0]->offset(0, 3, 0, 0);

		const Dtype* d1_t = target;
		const Dtype* d2_t = target + bottom[1]->offset(0, 1, 0, 0);
		const Dtype* d3_t = target + bottom[1]->offset(0, 2, 0, 0);
		const Dtype* d4_t = target + bottom[1]->offset(0, 3, 0, 0);

		Blob<Dtype> loss;
		loss.ReshapeLike(*bottom[2]);
		
		Dtype* loss_data = loss.mutable_gpu_data();
		caffe_gpu_set(loss.count(), Dtype(0), loss_data);
		
		IOULossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(count),
				CAFFE_CUDA_NUM_THREADS>>>(count, d1, d2, d3, d4, d1_t, d2_t, d3_t, d4_t, score_map,
				loss_data);
		
		caffe_gpu_asum(count, score_map, &valid_count_);
	
		Dtype loss_value;
		caffe_gpu_asum(count, loss_data, &loss_value);
		
		Dtype normalization = (valid_count_ != 0)?valid_count_:1.;
		top[0]->mutable_cpu_data()[0] = loss_value/normalization;
	}
	
	template <typename Dtype>
	void IOULossLayer<Dtype>::Backward_gpu(
			const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
			const vector<Blob<Dtype>*>& bottom) {

		if (propagate_down[0]) {
			
			const int count = bottom[0]->count(2);
			const Dtype* input_data = bottom[0]->gpu_data();
			const Dtype* target = bottom[1]->gpu_data();
			const Dtype* score_map = bottom[2]->gpu_data();
			Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

			Dtype* diff1 = bottom_diff;
			Dtype* diff2 = bottom_diff + bottom[0]->offset(0, 1, 0, 0);
			Dtype* diff3 = bottom_diff + bottom[0]->offset(0, 2, 0, 0);
			Dtype* diff4 = bottom_diff + bottom[0]->offset(0, 3, 0, 0);
			
			const Dtype* d1 = input_data;
			const Dtype* d2 = input_data + bottom[0]->offset(0, 1, 0, 0);
			const Dtype* d3 = input_data + bottom[0]->offset(0, 2, 0, 0);
			const Dtype* d4 = input_data + bottom[0]->offset(0, 3, 0, 0);

			const Dtype* d1_t = target;
			const Dtype* d2_t = target + bottom[1]->offset(0, 1, 0, 0);
			const Dtype* d3_t = target + bottom[1]->offset(0, 2, 0, 0);
			const Dtype* d4_t = target + bottom[1]->offset(0, 3, 0, 0);


			IOULossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(count),
				CAFFE_CUDA_NUM_THREADS>>>(count, d1, d2, d3, d4, d1_t, d2_t, d3_t, d4_t,
				diff1, diff2, diff3, diff4, score_map);
			
			// Scale down gradient
			Dtype normalization = (valid_count_ != 0)?valid_count_:1.;
			Dtype loss_weight = top[0]->cpu_diff()[0]/normalization;
			caffe_gpu_scal(bottom[0]->count(1), loss_weight, bottom_diff);
		}
	}
	
	INSTANTIATE_LAYER_GPU_FUNCS(IOULossLayer);
	
	}  // namespace caffe

