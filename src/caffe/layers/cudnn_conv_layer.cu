#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_conv_layer.hpp"

namespace caffe {

__global__ void sync_conv_groups() { }

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
// this->conv_iter_  是网络层参数，需要写入protxtx
/*
    // store input  保存初始相应层 的输入
    if ( this->conv_iter_ == 0)// && this->conv_id_ == 2)
    {
      const Dtype* bottom_data_c = bottom[0]->cpu_data();
      int bottom_size = bottom[0]->shape(0) *
        bottom[0]->shape(1) *
        bottom[0]->shape(2) *
        bottom[0]->shape(3);
      LOG(INFO) << "bottom_size: "<< bottom_size;
      //LOG(INFO) << "w1: "<< weight_c[0];

      char filename[12];
      sprintf(filename, "Oinput_%d", this->conv_id_);

      FILE * fp;
      fp = fopen(filename, "wb");
      if (fp !=NULL){
        fwrite(bottom_data_c, 4, bottom_size, fp);
      }
      fclose(fp);
    }
*/

/*
    // store weight and bias  保存 权重和 偏置
    if ( this->conv_iter_ == 0)// && this->conv_id_ == 2)
    {
      const Dtype* weight_c = this->blobs_[0]->cpu_data();
      int w_size = this->blobs_[0]->shape(0) *
        this->blobs_[0]->shape(1) *
        this->blobs_[0]->shape(2) *
        this->blobs_[0]->shape(3);
      LOG(INFO) << "w_size: "<< w_size;
      //LOG(INFO) << "w1: "<< weight_c[0];
    
      char filename[12];
      sprintf(filename, "Oconv_%d", this->conv_id_);
    
      FILE * fp;
      fp = fopen(filename, "wb");
      if (fp !=NULL){
        fwrite(weight_c, 4, w_size, fp);
      }
      fclose(fp);
    }
    if ( this->conv_iter_ == 0 && this->bias_term_)// && this->conv_id_ == 2)
    {
      const Dtype* bias_c = this->blobs_[1]->cpu_data();
      int b_size = this->blobs_[1]->shape(0);
      LOG(INFO) << "b_size: "<< b_size;
      //LOG(INFO) << "w1: "<< weight_c[0];
    
      char filename[12];
      sprintf(filename, "Obias_%d", this->conv_id_);
    
      FILE * fp;
      fp = fopen(filename, "wb");
      if (fp !=NULL){
        fwrite(bias_c, 4, b_size, fp);
      }
      fclose(fp);
    }
*/
      
    // Forward through cuDNN in parallel over groups.
    for (int g = 0; g < this->group_; g++) {
      // Filters.  卷积
      CUDNN_CHECK(cudnnConvolutionForward(handle_[g],
            cudnn::dataType<Dtype>::one,
            bottom_descs_[i], bottom_data + bottom_offset_ * g,
            filter_desc_, weight + this->weight_offset_ * g,
            conv_descs_[i],
            fwd_algo_[i], workspace[g], workspace_fwd_sizes_[i],
            cudnn::dataType<Dtype>::zero,
            top_descs_[i], top_data + top_offset_ * g));

      // Bias.  加上 偏置
      if (this->bias_term_) {
        const Dtype* bias_data = this->blobs_[1]->gpu_data();
        CUDNN_CHECK(cudnnAddTensor(handle_[g],
              cudnn::dataType<Dtype>::one,
              bias_desc_, bias_data + bias_offset_ * g,
              cudnn::dataType<Dtype>::one,
              top_descs_[i], top_data + top_offset_ * g));
      }
    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    sync_conv_groups<<<1, 1>>>();
      
	/*
    // store output  保存卷积后的输出===================
    if ( this->conv_iter_ == 0)// && this->conv_id_ == 2)
    {
      const Dtype* top_data_c = top[0]->cpu_data();
      int top_size = top[0]->shape(0) *
        top[0]->shape(1) *
        top[0]->shape(2) *
        top[0]->shape(3);
      LOG(INFO) << "top_size: "<< top_size;
      LOG(INFO) << "conv_id: : "<< this->conv_id_;
    
      char filename[12];
      sprintf(filename, "Otop_%d", this->conv_id_);
    
      FILE * fp;
      fp = fopen(filename, "wb");
      if (fp !=NULL){
        fwrite(top_data_c, 4, top_size, fp);
      }
      fclose(fp);
    }
*/
	
	
  }
  
  this->conv_iter_++;// 一开始为0，++
}

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = NULL;
  Dtype* weight_diff = NULL;
  if (this->param_propagate_down_[0]) {
    weight = this->blobs_[0]->gpu_data();
    weight_diff = this->blobs_[0]->mutable_gpu_diff();
  }
  Dtype* bias_diff = NULL;
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
  }
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Backward through cuDNN in parallel over groups and gradients.
    for (int g = 0; g < this->group_; g++) {
      // Gradient w.r.t. bias.
      if (this->bias_term_ && this->param_propagate_down_[1]) {
        CUDNN_CHECK(cudnnConvolutionBackwardBias(handle_[0*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              top_descs_[i],  top_diff + top_offset_ * g,
              cudnn::dataType<Dtype>::one,
              bias_desc_, bias_diff + bias_offset_ * g));
      }

      // Gradient w.r.t. weights.
      if (this->param_propagate_down_[0]) {
        const Dtype* bottom_data = bottom[i]->gpu_data();
        CUDNN_CHECK(cudnnConvolutionBackwardFilter(
              handle_[1*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              bottom_descs_[i], bottom_data + bottom_offset_ * g,
              top_descs_[i],    top_diff + top_offset_ * g,
              conv_descs_[i],
              bwd_filter_algo_[i], workspace[1*this->group_ + g],
              workspace_bwd_filter_sizes_[i],
              cudnn::dataType<Dtype>::one,
              filter_desc_, weight_diff + this->weight_offset_ * g));
      }

      // Gradient w.r.t. bottom data.
      if (propagate_down[i]) {
        if (weight == NULL) {
          weight = this->blobs_[0]->gpu_data();
        }
        Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
        CUDNN_CHECK(cudnnConvolutionBackwardData(
              handle_[2*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              filter_desc_, weight + this->weight_offset_ * g,
              top_descs_[i], top_diff + top_offset_ * g,
              conv_descs_[i],
              bwd_data_algo_[i], workspace[2*this->group_ + g],
              workspace_bwd_data_sizes_[i],
              cudnn::dataType<Dtype>::zero,
              bottom_descs_[i], bottom_diff + bottom_offset_ * g));
      }
    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    sync_conv_groups<<<1, 1>>>();
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNConvolutionLayer);

}  // namespace caffe
#endif
