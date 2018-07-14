
#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"// 包含各种头文件　blob，layer，net ,  solver 等
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/upgrade_proto.hpp"

#include "ristretto/quantization.hpp"

using caffe::Blob;   // 作为数据传输的媒介，无论是网络权重参数，还是输入数据，都是转化为Blob数据结构来存储
using caffe::Layer; // 作为网络的基础单元，神经网络中层与层间的数据节点、前后传递都在该数据结构中被实现
using caffe::Caffe;
using caffe::Net;    // 作为网络的整体骨架，决定了网络中的层次数目以及各个层的类别等信息
using caffe::Solver;// 作为网络的求解策略，涉及到求解优化问题的策略选择以及参数确定方面，修改这个模块的话一般都会是研究DL的优化求解的方向。
using caffe::shared_ptr;
using caffe::string;
using caffe::Timer;// 计时模块
using caffe::vector;
using std::ostringstream;

// 首先需要 使用 gflags的宏：DEFINE_xxxxx(变量名，默认值，help-string)  定义命令行参数
// 模型框架文件
DEFINE_string(model, "",
    "The model definition protocol buffer text file..");

// 圆模型权重
DEFINE_string(weights, "",
    "The trained weights.");

// 量化策略　动态固定点　迷你浮点  2方
DEFINE_string(trimming_mode, "",
    "Available options: dynamic_fixed_point, minifloat or "
    "integer_power_of_2_weights.");

// 输出保存的量化模型文件名
DEFINE_string(model_quantized, "",
    "The output path of the quantized net");

// gflags.h 参数 gpu序列
DEFINE_string(gpu, "",
    "Optional: Run in GPU mode on given device ID.");

// 量化迭代次数
DEFINE_int32(iterations, 1,
    "Optional: The number of iterations to run.");

// 最大量化误差范围 error_margin%
DEFINE_double(error_margin, 2,
    "Optional: the allowed accuracy drop in %");

// A simple registry for caffe commands.
typedef int (*BrewFunction)();// 函数指针
typedef std::map<caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;//g_brew_map 函数名与函数指针键值对map映射
//  　不同函数的注册类 将函数 记录到 g_brew_map 
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

// 由函数名字符串获取 执行对应应函数指针的函数==========================
static BrewFunction GetBrewFunction(const caffe::string& name) {
  if (g_brew_map.count(name)) {//查看记录的键值对映射中是否有该种类型的层名字
    return g_brew_map[name];// 返回对应函数名字的函数指针 并执行相应的函数
  } 
  else 
  {
      LOG(ERROR) << "Available ristretto actions:";
      for (BrewMap::iterator it = g_brew_map.begin();
	  it != g_brew_map.end(); ++it) {
	LOG(ERROR) << "\t" << it->first;// 打印支持的层名字
      }
      LOG(FATAL) << "Unknown action: " << name;//　打印对应的不存在的 层名字
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
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";// 原模型框架
  CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";// 原模型权重
  CHECK_GT(FLAGS_model_quantized.size(), 0) << "Need network description "  "output path.";// 量化输出的模型框架文件
  CHECK_GT(FLAGS_trimming_mode.size(), 0) << "Need trimming mode.";// 量化策略
  // 定义量化对象
  Quantization* q = new Quantization(FLAGS_model, FLAGS_weights,
      FLAGS_model_quantized, FLAGS_iterations, FLAGS_trimming_mode,
      FLAGS_error_margin, FLAGS_gpu);
  // 执行量化
  q->QuantizeNet();
  
  delete q;
  return 0;
}
// 注册该函数到 g_brew_map (函数名与函数指针键值对map映射)
RegisterBrewFunction(quantize);

int main(int argc, char** argv) {
  // Print output to stderr (while still logging).
  FLAGS_alsologtostderr = 1;
  // Set version
  gflags::SetVersionString(AS_STRING(CAFFE_VERSION));
  // Usage message.
   // 工具命令行参数
  gflags::SetUsageMessage("command line brew\n"
      "usage: ristretto <command> <args>\n\n"
      "commands:\n"
      "  quantize        Trim 32bit floating point net\n");
  // Run tool or show usage.
  caffe::GlobalInit(&argc, &argv);
  if (argc == 2) {
      return GetBrewFunction(caffe::string(argv[1]))();// 根据命令行参数执行相应函数
  }
  else {
      gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/ristretto");
  }
}
