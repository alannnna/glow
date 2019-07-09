/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/Importer/ONNXModelLoader.h"
#include "glow/Support/Support.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"

#include <glog/logging.h>

#include <fstream>

using namespace glow;

namespace {
llvm::cl::OptionCategory onnxCat("ONNX Runner Options");
llvm::cl::opt<std::string> executionBackend(
    "backend",
    llvm::cl::desc("Backend to use, e.g., Interpreter, CPU, OpenCL:"),
    llvm::cl::Optional, llvm::cl::init("Interpreter"), llvm::cl::cat(onnxCat));
} // namespace

static void runONNX(std::string fileName,
                    llvm::ArrayRef<size_t> inputShape,
                    std::string input_name) {
  // Get the EE and the fn
  ExecutionEngine EE{};
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  
  PlaceholderBindings bindings;
  Placeholder *graphOutputVar;
  Type input_type(ElemKind::FloatTy, inputShape);
  
  ONNXModelLoader onnxLD(fileName, {input_name.c_str()}, {&input_type}, *F);

  // Make an input tensor
  graphOutputVar = EXIT_ON_ERR(onnxLD.getSingleOutput());
  auto PH = mod.getPlaceholderByName(input_name);
  auto *inTensor = bindings.allocate(PH);
  inTensor->getHandle().randomize(-10.0, 10.0, mod.getPRNG());

  // Compile and run the graph, and print the output
  EE.compile(CompilationMode::Infer, F);
  EE.run(bindings);
  auto result = bindings.get(graphOutputVar)->getHandle();
  result.dump();
}

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, " The MNIST test\n\n");
  
  runONNX("/Users/atem/gitrepos/differentiable/scala/model.onnx",
          {1, 2}, "x");

  return 0;
}
