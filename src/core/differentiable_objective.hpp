#pragma once
#include <torch/torch.h>
#include <vector>
#include <string>

namespace DSO {
class DifferentiableObjective {
    public:
        virtual ~DifferentiableObjective() = default;
        virtual torch::Tensor forward() = 0;
        virtual std::vector<torch::Tensor> parameters() = 0;
        virtual const std::vector<std::string>& parameter_names() const = 0;
};
} // namespace DSO