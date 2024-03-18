#include "pixelflow/core/Indexer.h"

namespace pixelflow {
namespace core {

    Indexer::Indexer(const std::vector<Image> &input_images,
                     const Image &output_image,
                     DtypePolicy dtype_policy,
                     const ShapeArray &reduction_dims) {

    }

    Indexer::Indexer(const std::vector<Image> &input_images,
                     const std::vector<Image> &output_images,
                     DtypePolicy dtype_policy,
                     const ShapeArray &reduction_dims) {

        num_inputs_ = static_cast<int64_t>(input_images.size());
        num_outputs_ = static_cast<int64_t>(output_images.size());

        // Checking number of inputs and outputs
        if (num_inputs_ < 1) {
            LogError("Indexer must have at least one input.");
        }

        if (num_inputs_ > MAX_INPUTS) {
            std::ostringstream oss;
            oss << "Indexer cannot have more than " << MAX_INPUTS
            << " inputs and got " << num_inputs_;
            LogError(oss.str().c_str());
        }
        if (num_outputs_ < 1) {
            LogError("Indexer must have at least one output.");
        }
        if (num_outputs_ > MAX_OUTPUTS) {
            std::ostringstream oss;
            oss << "Indexer cannot have more than " << MAX_OUTPUTS
            << " outputs and got " << num_outputs_;
            LogError(oss.str().c_str());
        }

        // Check DtypePolicy.
        if (dtype_policy == DtypePolicy::ALL_SAME) {
            const PfType ref_dtype = input_images[0].GetDtype();
            for (const auto& input_image : input_images) {
                if (input_image.GetDtype() != ref_dtype) {
                    std::ostringstream oss;
                    oss << "Dype mismatch input " << input_image.GetDtype().ToString()
                    << " != " << ref_dtype.ToString();
                    LogError(oss.str().c_str());
                }
            }
            for (const auto& output_image : output_images) {
                if (output_image.GetDtype() != ref_dtype) {
                    std::ostringstream oss;
                    oss << "Dype mismatch output " << output_image.GetDtype().ToString()
                    << " != " << ref_dtype.ToString();
                    LogError(oss.str().c_str());
                }
            }
        } else if (dtype_policy == DtypePolicy::INPUT_SAME) {
            const PfType ref_dtype = input_images[0].GetDtype();
            for (const auto& input_image : input_images) {
                if (input_image.GetDtype() != ref_dtype) {
                    std::ostringstream oss;
                    oss << "Dype mismatch between inputs " << input_image.GetDtype().ToString()
                    << " != " << ref_dtype.ToString();
                    LogError(oss.str().c_str());
                }
            }
        } else if (dtype_policy == DtypePolicy::INPUT_SAME_OUTPUT_BOOL) {
            const PfType ref_dtype = input_images[0].GetDtype();
            for (const auto& input_image : input_images) {
                if (input_image.GetDtype() != ref_dtype) {
                    std::ostringstream oss;
                    oss << "Dype mismatch between inputs " << input_image.GetDtype().ToString()
                    << " != " << ref_dtype.ToString();
                    LogError(oss.str().c_str());
                }
            }
            for (const auto& output_image : output_images) {
                if (output_image.GetDtype() != PfType::Bool) {
                    std::ostringstream oss;
                    oss << "Dype mismatch between outputs " << output_image.GetDtype().ToString()
                    << " != " << ref_dtype.ToString();
                    LogError(oss.str().c_str());
                }
            }
        } else if (dtype_policy == DtypePolicy::NONE) {
            // Do nothing.
        } else {
            LogError("Unimplemented dtype policy");
        }
    }



} // namespace core
}// namespace pixelflow