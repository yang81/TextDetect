/******************************************************

	File Name       : online_east_target_layer.cu
	File Description: A file
	Author          : yanghongyu
	Create Time     : 2020-4-4 17:20:25

*******************************************************/


#include "caffe/layers/online_east_target_layer.hpp"
#include "caffe/layer.hpp"


namespace caffe {

__device__ void target_inRectangle_border_distance_gpu(float xc, float yc, float width, float height, float alpha, float x, float y, EastInTarget &it){
	float eps = 0.00000000000001;
	float pi =  3.14159265358;
	
	float shrink_h = (height < 16)?0.3:0.3;
	float shrink_w = (width < 16)?0.3:0.3;

	float norm1 = 0;
	float norm2 = 0;

	float min_side = (width < height)?width:height;
	
	//width = width - 2 * shrink_w * width;
	//height = height - 2 * shrink_h * height;

	//width = width - 2 * shrink_w * min_side;
	//height = height - 2 * shrink_w * min_side;
	
	width /= 2.;
	height /= 2.;

	it.inRectangle = false;
	it.d1 = 0;
	it.d2 = 0;
	it.d3 = 0;
	it.d4 = 0;


	if (fabs(alpha - 0) < eps){
		
		//std::cout<<"+++++++++++++++++++++++++++++++++++"<<std::endl;
		if (x >= xc - width && x <= xc + width && y >= yc - height && y <= yc + height){
			it.inRectangle = true;

			it.d1 = fabs(y - yc - height);
			it.d3 = fabs(y - yc + height);
			it.d2 = fabs(x - xc - width);
			it.d4 = fabs(x - xc + width);

			it.border_distance = min(min(it.d1, it.d3),min(it.d2, it.d4));
			
			return;
		}
		else{
			
			float d1 = fabs(y - yc - height);
			float d3 = fabs(y - yc + height);
			float d2 = fabs(x - xc - width);
			float d4 = fabs(x - xc + width);
			
			//region1,3
			if (x > xc - width && x < xc + width){
				it.border_distance = min(d1, d3);
			}

			//region2,4
			if (y > yc - height && y < yc + height){
				it.border_distance = min(d2, d4);
			}

			//region5
			if (x > xc + width && y > yc + height){
				it.border_distance = sqrt(d1*d1 + d2*d2);
			}

			//region6
			if (x > xc + width && y < yc - height){
				it.border_distance = sqrt(d2*d2 + d3*d3);
			}

			//region7
			if (x < xc - width && y < yc - height){
				it.border_distance = sqrt(d3*d3 + d4*d4);
			}

			//region8
			if (x < xc - width && y > yc + height){
				it.border_distance = sqrt(d1*d1 + d4*d4);
			}
		
			return;
		}
			
	}
	
	if (fabs(fabs(alpha) - pi/2.) < eps){
		
		//std::cout<<"***************************************"<<std::endl;
		if (x >= xc - width && x <= xc + width && y >= yc - height && y <= yc + height){
			it.inRectangle = true;

			it.d1 = fabs(x - xc + height);
			it.d3 = fabs(x - xc - height);
			it.d2 = fabs(y - yc - width);
			it.d4 = fabs(y - yc + width);

			it.border_distance = min(min(it.d1, it.d3),min(it.d2, it.d4));

			return;
		}else{
				
			float d1 = fabs(x - xc + height);
			float d3 = fabs(x - xc - height);
			float d2 = fabs(y - yc - width);
			float d4 = fabs(y - yc + width);
			
			//region1,3
			if (y > yc - width && y < yc + width){
				it.border_distance = min(d1, d3);
			}
		
			//region2,4
			if (x > xc - height && x < xc + height){
				it.border_distance = min(d2, d4);
			}
		
			//region5
			if (y < yc + width && x < xc - height){
				it.border_distance = sqrt(d1*d1 + d2*d2);
			}
		
			//region6
			if (y < yc + width && x > xc + height){
				it.border_distance = sqrt(d2*d2 + d3*d3);
			}
		
			//region7
			if (y < yc - width && x > xc + height){
				it.border_distance = sqrt(d3*d3 + d4*d4);
			}
		
			//region8
			if (y < yc - width && x < xc - height){
				it.border_distance = sqrt(d1*d1 + d4*d4);
			}
		
			return;
		}

	}
	
	norm1 = width/sqrt(1. + tan(alpha)* tan(alpha));
	norm2 = height/sqrt(1. + 1./(tan(alpha)* tan(alpha)));
	
	float x2 = xc + norm1;
	float x4 = xc - norm1;
	float y2 = tan(alpha)*(x2 - xc) + yc;
	float y4 = tan(alpha)*(x4 - xc) + yc;
	
	float x1 = xc - norm2;
	float x3 = xc + norm2;
	float y1 = -1./tan(alpha)*(x1 - xc) + yc;
	float y3 = -1./tan(alpha)*(x3 - xc) + yc;

	float d2 = abs(x + tan(alpha)*y - x2 - tan(alpha)*y2)/sqrt(1 + tan(alpha)*tan(alpha));
	float d4 = abs(x + tan(alpha)*y - x4 - tan(alpha)*y4)/sqrt(1 + tan(alpha)*tan(alpha));
	float d1 = abs(tan(alpha)*x - y + y1 - tan(alpha)*x1)/sqrt(1 + tan(alpha)*tan(alpha));
	float d3 = abs(tan(alpha)*x - y + y3 - tan(alpha)*x3)/sqrt(1 + tan(alpha)*tan(alpha));

	bool l1 = false;
	bool l2 = false;
	bool l3 = false;
	bool l4 = false;

	bool round_b = false;
	if (round_b){
		if (round((-1./tan(alpha)*(xc - x2) + y2 - yc)*(-1./tan(alpha)*(x - x2) + y2 - y)) >= 0)
			l2 = true;
		
		if (round((-1./tan(alpha)*(xc - x4) + y4 - yc)*(-1./tan(alpha)*(x - x4) + y4 - y)) >= 0)
			l4 = true;

		if (round((tan(alpha)*(xc - x1) + y1 - yc)*(tan(alpha)*(x - x1) + y1 - y)) >= 0)
			l1 = true;

		if (round((tan(alpha)*(xc - x3) + y3 - yc)*(tan(alpha)*(x - x3) + y3 - y)) >= 0)
			l3 = true;
	}else{
		if ((-1./tan(alpha)*(xc - x2) + y2 - yc)*(-1./tan(alpha)*(x - x2) + y2 - y) >= 0)
			l2 = true;
		
		if ((-1./tan(alpha)*(xc - x4) + y4 - yc)*(-1./tan(alpha)*(x - x4) + y4 - y) >= 0)
			l4 = true;

		if ((tan(alpha)*(xc - x1) + y1 - yc)*(tan(alpha)*(x - x1) + y1 - y) >= 0)
			l1 = true;

		if ((tan(alpha)*(xc - x3) + y3 - yc)*(tan(alpha)*(x - x3) + y3 - y) >= 0)
			l3 = true;
	}



	if (l1&&l2&&l3&&l4){
		
		it.inRectangle = true;
		
		it.d1 = d1;
		it.d2 = d2;
		it.d3 = d3;
		it.d4 = d4;

		it.border_distance = min(min(d1, d3),min(d2, d4));
	}else{

		//region 1,3
		if (l2&&l4){
			it.border_distance = min(d1, d3);
		}

		//region 2,4
		if (l1&&l3){
			it.border_distance = min(d2, d4);
		}

		//region 5
		if (!(l1&&l2)){
			it.border_distance = sqrt(d1*d1 + d2*d2);
		}

		//region 6
		if (!(l2&&l3)){
			it.border_distance = sqrt(d2*d2 + d3*d3);
		}

		//region 7
		if (!(l3&&l4)){
			it.border_distance = sqrt(d3*d3 + d4*d4);
		}

		//region 8
		if (!(l4&&l1)){
			it.border_distance = sqrt(d4*d4 + d1*d1);
		}

	}

	//std::cout<<"------------------------------"<<std::endl;
	return;
}


template <typename Dtype>
__global__ void TargetForward_GPU(const int nthreads,
Dtype* bboxes, int count_h, int count_w, int w_, int h_,
Dtype* score_map, Dtype* score_weight_map, Dtype* geometry_map, Dtype* geometry_weight, Dtype* angle_map, Dtype* angle_weight,
Dtype* tcbp_map, Dtype* tcbp_weight_map, int sample_factor, float abstract_area_down, float abstract_area_up){

	CUDA_KERNEL_LOOP(i, nthreads) {
		int h = i/w_;
		int w = i%w_;

		Dtype s[50]={0};
		Dtype s_sum = 0;
		Dtype s_min = FLT_MAX;
		int s_min_index = 0;
		Dtype s_positive = 0;
		Dtype s_negative = 0;
		
		for (int b = 0; b < count_h; ++b) {			
			Dtype xc = bboxes[count_w*b + 0];
			Dtype yc = bboxes[count_w*b + 1];
			Dtype width = bboxes[count_w*b + 2];
			Dtype height = bboxes[count_w*b + 3];
			Dtype alpha = bboxes[count_w*b + 4];

			s_positive += width*height;
			
		}

		s_negative = 512*512 - s_positive;
				
		for (int b = 0; b < count_h; ++b) {
			
			EastInTarget it;
			
			Dtype xc = bboxes[count_w*b + 0];
			Dtype yc = bboxes[count_w*b + 1];
			Dtype width = bboxes[count_w*b + 2];
			Dtype height = bboxes[count_w*b + 3];
			Dtype alpha = bboxes[count_w*b + 4];

			/*
			Dtype lx = bboxes[5*b + 5];
			Dtype ly = bboxes[5*b + 6];
			Dtype rx = bboxes[5*b + 7];
			Dtype ry = bboxes[5*b + 8];
			*/

			target_inRectangle_border_distance_gpu(xc, -yc, width, height, alpha, w * sample_factor, -h * sample_factor, it);

			if (it.inRectangle){
				
				if (score_map){
					score_map[i] = 1;
				}

				if (score_weight_map){		
					if (it.border_distance < 2){
						score_weight_map[w_ * h_ * 0 + i] = 0.1*s_negative/(512*512);
					}else{
						score_weight_map[w_ * h_ * 0 + i] = s_negative/(512*512);
					}
				}
				
				
				if (geometry_map != NULL){
					geometry_map[w_ * h_ * 0 + i] = it.d1;
					geometry_map[w_ * h_ * 1 + i] = it.d2;
					geometry_map[w_ * h_ * 2 + i] = it.d3;
					geometry_map[w_ * h_ * 3 + i] = it.d4;
				}

				if (geometry_weight != NULL){
					geometry_weight[w_ * h_ * 0 + i] = 1;
					geometry_weight[w_ * h_ * 1 + i] = 1;
					geometry_weight[w_ * h_ * 2 + i] = 1;
					geometry_weight[w_ * h_ * 3 + i] = 1;
				}

				/*
				if (geometry_map != NULL){
					geometry_map[w_ * h_ * 0 + i] = (w - lx/sample_factor)/height;
					geometry_map[w_ * h_ * 1 + i] = (h - ly/sample_factor)/height;
					geometry_map[w_ * h_ * 2 + i] = (w - rx/sample_factor)/height;
					geometry_map[w_ * h_ * 3 + i] = (h - ry/sample_factor)/height;
					geometry_map[w_ * h_ * 4 + i] = height/sample_factor/sample_factor/sample_factor;
				}

				if (geometry_weight != NULL){
					geometry_weight[w_ * h_ * 0 + i] = 1;
					geometry_weight[w_ * h_ * 1 + i] = 1;
					geometry_weight[w_ * h_ * 2 + i] = 1;
					geometry_weight[w_ * h_ * 3 + i] = 1;
					geometry_weight[w_ * h_ * 4 + i] = 1;
				}*/
								
				
				if (angle_map != NULL){
					angle_map[w_ * h_ * 0 + i] = alpha;
				}

				if (angle_weight != NULL){
					angle_weight[w_ * h_ * 0 + i] = 1;
				}

				
				if (tcbp_map != NULL){
					tcbp_map[w_ * h_ * 0 + i] = it.border_distance * 2./height;	
				}

				if (tcbp_weight_map){								
					if (it.border_distance < 2){
						tcbp_weight_map[w_ * h_ * 0 + i] = 10;
					}else{
						tcbp_weight_map[w_ * h_ * 0 + i] = 1;
					}
				}

			}else{
					
					if (score_weight_map){
						if (it.border_distance < 3){
							score_weight_map[w_ * h_ * 0 + i] = 10*s_positive/(512*512);
						}else{
							score_weight_map[w_ * h_ * 0 + i] = 3*s_positive/(512*512);
						}
					}					
			}

		}
	}
}



template <typename Dtype>
void OnlineEastTargetLayer<Dtype>::Forward_gpu(
  const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {

	Dtype* bboxs_data_gpu = bottom[1]->mutable_gpu_data();
	
	
	Dtype* score_data = NULL;
	Dtype* score_weight_data = NULL;
	Dtype* geometry_data = NULL;
	Dtype* geometry_weight = NULL;
	Dtype* angle_data = NULL;
	Dtype* angle_weight = NULL; 
	Dtype* tcbp_data = NULL;
	Dtype* tcbp_weight = NULL;


	for (int i = 0; i < blob_names_.size(); ++i) {
		string blob_name = blob_names_[i];
		
		if (blob_name == "score_label"){
			score_data = top[i]->mutable_gpu_data();
  		caffe_gpu_set(top[i]->count(), Dtype(0), score_data);
		}


		if (blob_name == "score_weight"){
			score_weight_data = top[i]->mutable_gpu_data();
			caffe_gpu_set(top[i]->count(), Dtype(0), score_weight_data);
		}
		
		if (blob_name == "geometry_label"){
			geometry_data = top[i]->mutable_gpu_data();
			caffe_gpu_set(top[i]->count(), Dtype(0), geometry_data);
		}

		if (blob_name == "geometry_weight"){
			geometry_weight = top[i]->mutable_gpu_data();
			caffe_gpu_set(top[i]->count(), Dtype(0), geometry_weight);
		}

		if (blob_name == "angle_label"){
			angle_data = top[i]->mutable_gpu_data();
			caffe_gpu_set(top[i]->count(), Dtype(0), angle_data);
		}

		if (blob_name == "angle_weight"){
			angle_weight = top[i]->mutable_gpu_data();
			caffe_gpu_set(top[i]->count(), Dtype(0), angle_weight);
		}

		if (blob_name == "tcbp_label"){
			tcbp_data = top[i]->mutable_gpu_data();
			caffe_gpu_set(top[i]->count(), Dtype(0), tcbp_data);
		}

		if (blob_name == "tcbp_weight"){
			tcbp_weight = top[i]->mutable_gpu_data();
			caffe_gpu_set(top[i]->count(), Dtype(0), tcbp_weight);
		}
	}

	
	Dtype valid_count_ = 0;
	const int count = bottom[0]->count(2);	
	
	TargetForward_GPU<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
		(count, bboxs_data_gpu, bottom[1]->height(),bottom[1]->width(), bottom[0]->width(), bottom[0]->height(), 
		score_data, score_weight_data, geometry_data, geometry_weight, angle_data, angle_weight,  tcbp_data, tcbp_weight, 
		sample_factor_, abstract_area_down_, abstract_area_up_);
	
}

template <typename Dtype>
void OnlineEastTargetLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
    const vector<Blob<Dtype> *> &bottom) {

	for (int i = 0; i < propagate_down.size(); ++i) {
    if (propagate_down[i]) {
      NOT_IMPLEMENTED;
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(OnlineEastTargetLayer);

} // namespace caffe


