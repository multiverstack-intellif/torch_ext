// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include "torch/extension.h"
#include "nms.h"

static auto registry = torch::RegisterOperators()
  .op("opt_ops::static_nms", &static_nms)
  .op("opt_ops::static_batched_nms", &static_batched_nms);
