// 模型 量化 工具============================
#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/upgrade_proto.hpp"

#include "ristretto/quantization.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::Solver;
using caffe::shared_ptr;
using caffe::string;
using caffe::Timer;
using caffe::vector;
using std::ostringstream;

//首先需要 使用 gflags的宏：DEFINE_xxxxx(变量名，默认值，help-string)  定义命令行参数=========================
// 使用 google::ParseCommandLineFlags(&argc, &argv, true); 解析后
// 就可以使用 FLAGS_  变量名访问对应的命令行参数了
// 例如 FLAGS_model    FLAGS_weights    FLAGS_trimming_mode   FLAGS_gpu    FLAGS_iterations   FLAGS_error_margin

// gflags 定义命令行会出现的参数
DEFINE_string(model, "",
    "The model definition protocol buffer text file..");
DEFINE_string(weights, "",
    "The trained weights.");
DEFINE_string(trimming_mode, "",
    "Available options: dynamic_fixed_point, minifloat or "
    "integer_power_of_2_weights.");
DEFINE_string(model_quantized, "",
    "The output path of the quantized net");
DEFINE_string(gpu, "",
    "Optional: Run in GPU mode on given device ID.");
DEFINE_int32(iterations, 50,
    "Optional: The number of iterations to run.");
DEFINE_double(error_margin, 2,
    "Optional: the allowed accuracy drop in %");


// ========================== add net type ================
// 分类网络/检测网络(classification /ssd_detection / yolov2_detection / faster_rcnn_detection)
DEFINE_string(net_type, "", "The net type: classification /ssd_detection / yolov2_detection");
/////////////////////////////////////////


// A simple registry for caffe commands.
typedef int (*BrewFunction)();// 函数指针
typedef std::map<caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;// 函数名字: 函数指针 map映射键值对

// 注册已有函数进  map映射键值对 g_brew_map
#define RegisterBrewFunction(func) \
namespace { \
class __Registerer_##func { \
 public: /* NOLINT */ \
  __Registerer_##func() { \
    g_brew_map[#func] = &func; \
  } \
}; \
__Registerer_##func g_registerer_##func; \
}

static BrewFunction GetBrewFunction(const caffe::string& name) {
  if (g_brew_map.count(name)) {
    return g_brew_map[name];// 返回命令行对于的函数指针，并执行
  } else {
    LOG(ERROR) << "Available ristretto actions:";
    for (BrewMap::iterator it = g_brew_map.begin();
         it != g_brew_map.end(); ++it) {
      LOG(ERROR) << "\t" << it->first;// 打印支持的函数
    }
    LOG(FATAL) << "Unknown action: " << name;
    return NULL;  // not reachable, just to suppress old compiler warnings.
  }
}

// ristretto commands to call by
//     ristretto <command> <args>
//
// To add a command, define a function "int command()" and register it with
// RegisterBrewFunction(action);

// Quantize a 32-bit FP network to smaller word width.
int quantize(){
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";// 输入原来未量化的模型框架文件名
  CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";// 对应的权重文件
  CHECK_GT(FLAGS_model_quantized.size(), 0) << "Need network description "// 量化输出的模型框架文件
      "output path.";
  CHECK_GT(FLAGS_trimming_mode.size(), 0) << "Need trimming mode.";// 量化策略
  CHECK_GT(FLAGS_net_type.size(), 0) << "Need define net type.";// 网络模型 默认ssd_dection
  
  
  Quantization* q = new Quantization(FLAGS_model, FLAGS_weights,
      FLAGS_model_quantized, FLAGS_iterations, FLAGS_trimming_mode,
      FLAGS_error_margin, FLAGS_gpu, FLAGS_net_type);// 量化对象
 
  q->QuantizeNet();// 执行量化
  delete q;
  return 0;
}
RegisterBrewFunction(quantize);// 注册该函数进 map映射键值对 g_brew_map

int main(int argc, char** argv) {
  // Print output to stderr (while still logging).
  FLAGS_alsologtostderr = 1;
  // Set version
  gflags::SetVersionString(AS_STRING(CAFFE_VERSION));//版本
  // Usage message.
  // 使用信息
  gflags::SetUsageMessage("command line brew\n"
      "usage: ristretto <command> <args>\n\n"
      "commands:\n"
      "  quantize        Trim 32bit floating point net\n");
  // Run tool or show usage.
  caffe::GlobalInit(&argc, &argv);// gflags 命令行参数初始化， glog 段错误捕捉初始化
  if (argc == 2) {
      return GetBrewFunction(caffe::string(argv[1]))();// 执行命令行对应的函数
  } else {
      gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/ristretto");//显示使用信息
  }
}
