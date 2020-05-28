/******************************************************

	File Name       : online_east_target_layer.cpp
	File Description: A file
	Author          : yanghongyu
	Create Time     : 2020-4-4 17:20:59

*******************************************************/

#include "caffe/layers/online_east_target_layer.hpp"


namespace caffe {

using std::vector;

template <typename Dtype>
void OnlineEastTargetLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){
	TargetParameter target_param = this->layer_param_.target_param();

	sample_factor_ = target_param.sample_factor();
	abstract_area_down_ = target_param.abstract_area_down();
	abstract_area_up_ = target_param.abstract_area_up();
	
	blob_names_.clear();
	
	for (int i = 0; i < target_param.blob_name_size(); ++i) {
		string blob_name = target_param.blob_name(i);
		blob_names_.push_back(blob_name);
	}
}

template <typename Dtype>
void OnlineEastTargetLayer<Dtype>::Reshape(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	for (int i = 0; i < blob_names_.size(); ++i) {
		string blob_name = blob_names_[i];

		if (blob_name == "score_label"){
			top[i]->Reshape(1, 1, bottom[0]->height(), bottom[0]->width());
		}


		if (blob_name == "score_weight"){
			top[i]->Reshape(1, 1, bottom[0]->height(), bottom[0]->width());
		}

	
		if (blob_name == "geometry_label"){
			top[i]->Reshape(1, 4, bottom[0]->height(), bottom[0]->width());
			//top[i]->Reshape(1, 5, bottom[0]->height(), bottom[0]->width());
		}

		if (blob_name == "geometry_weight"){
			top[i]->Reshape(1, 4, bottom[0]->height(), bottom[0]->width());
			//top[i]->Reshape(1, 5, bottom[0]->height(), bottom[0]->width());
		}

		if (blob_name == "angle_label"){
			top[i]->Reshape(1, 1, bottom[0]->height(), bottom[0]->width());
		}

		if (blob_name == "angle_weight"){
			top[i]->Reshape(1, 1, bottom[0]->height(), bottom[0]->width());
		}

		if (blob_name == "tcbp_label"){
			top[i]->Reshape(1, 1, bottom[0]->height(), bottom[0]->width());
		}

		if (blob_name == "tcbp_weight"){
			top[i]->Reshape(1, 1, bottom[0]->height(), bottom[0]->width());
		}
	}

}


template <typename Dtype>
void OnlineEastTargetLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  
	NOT_IMPLEMENTED;
}

template <typename Dtype>
void OnlineEastTargetLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
    const vector<Blob<Dtype> *> &bottom) {

	NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(OnlineEastTargetLayer);
#endif

INSTANTIATE_CLASS(OnlineEastTargetLayer);
REGISTER_LAYER_CLASS(OnlineEastTarget);

} // namespace caffe


