#include <array>
#include <onnxruntime_cxx_api.h>

#include <iostream>
#include <vector>

int main(int argc, char *argv[]) {
  std::string model_path = argv[1];

  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

  std::array<int64_t, 2> input_shape{1, 10};
  std::vector<int32_t> input_ids{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<int32_t> attention_mask{1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<int32_t> token_type_ids{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  std::array<int64_t, 2> output_shape_{1, 384};
  std::vector<std::array<float, 384>> results{{}};

  Ort::Value input_ids_tensor = Ort::Value::CreateTensor<int32_t>(
      memory_info, input_ids.data(), input_ids.size(), input_shape.data(),
      input_shape.size());
  Ort::Value attention_mask_tensor = Ort::Value::CreateTensor<int32_t>(
      memory_info, attention_mask.data(), attention_mask.size(),
      input_shape.data(), input_shape.size());
  Ort::Value token_type_ids_tensor = Ort::Value::CreateTensor<int32_t>(
      memory_info, token_type_ids.data(), token_type_ids.size(),
      input_shape.data(), input_shape.size());

  Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
      memory_info, (float *)results.data(), results.size() * 384,
      output_shape_.data(), output_shape_.size());

  Ort::Env env;
  std::cout << "Creating session" << std::endl;
  Ort::Session session{env, model_path.c_str(), Ort::SessionOptions{nullptr}};

  const char *input_names[] = {"input_ids", "attention_mask", "token_type_ids"};
  const char *output_names[] = {"output"};

  Ort::RunOptions run_options;
  std::array<Ort::Value, 3> input_tensors{std::move(input_ids_tensor),
                                          std::move(attention_mask_tensor),
                                          std::move(token_type_ids_tensor)};
  std::cout << "Running session" << std::endl;
  session.Run(run_options, input_names, input_tensors.data(), 3, output_names,
              &output_tensor, 1);

  std::cout << "Output: " << std::endl;
  std::cout << "Output length: " << results[0].size() << std::endl;
  for (int i = 0; i < 384; i++) {
    std::cout << results[0][i] << " ";
  }
}