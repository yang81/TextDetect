/******************************************************

	File Name       : online_east_target_layer.hpp
	File Description: A file
	Author          : yanghongyu
	Create Time     : 2020-4-4 17:21:18

*******************************************************/


#ifndef CAFFE_ONLINE_EAST_TARGET_LAYER_HPP_
#define CAFFE_ONLINE_EAST_TARGET_LAYER_HPP_

#include <vector>
#include <float.h>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {

template <typename Dtype>
class OnlineEastTargetLayer : public Layer<Dtype> {
 public:
  explicit OnlineEastTargetLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "OnlineEastTarget"; }

  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int MinTopBlobs() const { return 2; }
  virtual inline int MaxTopBlobs() const { return 8; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	std::vector<string> blob_names_;
	int sample_factor_;
	float abstract_area_down_;
	float abstract_area_up_;

};

struct EastInTarget{
	bool inRectangle;
	
	float d1,d2,d3,d4;
	
	float border_distance;
};

}  // namespace caffe

#endif


