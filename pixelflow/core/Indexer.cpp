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

        // Convert inputs and outputs to ImageRef
        for (int64_t i = 0; i != num_inputs_; ++i) {
            inputs_[i] = ImageRef(input_images[i]);
        }
        for (int64_t i = 0; i != num_outputs_; ++i) {
            outputs_[i] = ImageRef(output_images[i]);
        }

        // For simplicity, all outputs must have the same shape.
        ShapeArray ref_output_shape = output_images[0].Shape();
        for (const auto& output_image : output_images) {
            if (output_image.Shape() != ref_output_shape) {
                std::ostringstream oss;
                oss << "For broadcast, all output shapes must be the same, "
                << "but " << output_image.Shape().ToString()
                << " != " << ref_output_shape.ToString();
                LogError(oss.str().c_str());
            }
        }

        // Theoretically, reduction can be mixed with broadcasting. For
        // simplicity, we require explicit broadcasting after reduction.
        if (reduction_dims.size() > 0) {
            if (num_inputs_ != 1) {
                LogError("Internal error: reduction op can only have 1 inputs.");
            }

            for (int64_t i = 0; i < num_outputs_; ++i) {
                // Sanity check. The indexer only handles keepdim == true.
                // This also ensures that reduction is not mixed with broadcasting.
                if (ReductionShape(input_images[0].Shape(),
                                               reduction_dims, true) !=
                    output_images[i].Shape()) {
                    std::ostringstream oss;
                    oss << "Reduction dimensions mismatch, input's shape "
                    << input_images[0].Shape().ToString() << "reduction dims "
                    << reduction_dims.ToString() << " output's shape "
                    << output_images[i].Shape().ToString();
                    LogError(oss.str().c_str());
                    }

                // For each reduction dim, set the corresponding output strides to
                // 0.
                ReductionRestride(outputs_[i], inputs_[0].ndims_, inputs_[0].shape_,
                                  reduction_dims);
            }

            // ndims_ == inputs_[0].ndims_ == output_.ndims
            ndims_ = inputs_[0].ndims_;
        }
    }

    void Indexer::ReductionRestride(ImageRef& dst,
                                int64_t src_ndims,
                                const int64_t* src_shape,
                                const ShapeArray& reduction_dims) {
        if (dst.ndims_ != src_ndims) {
            std::ostringstream oss;
            oss << "Internal error, src ndims: " << src_ndims
            << " != " << dst.ndims_;
            LogError(oss.str().c_str());
        }
        for (int64_t i = 0; i < dst.ndims_; ++i) {
            if (dst.shape_[i] == 1 && src_shape[i] != 1) {
                dst.byte_strides_[i] = 0;
            }
        }
    }

    void Indexer::ReorderDimensions(const ShapeArray& reduction_dims) {
        if (ndims_ == 1) {
            return;
        }

        ShapeArray permute(ndims_);
        std::iota(permute.rbegin(), permute.rend(), 0);

        // Returns -1 / 0 / 1 indicates no_swap / tbd / swap dim0 with dim1.
        auto ShouldSwap = [&](size_t dim0, size_t dim1) {
            // Outputs
            for (int64_t i = 0; i < num_outputs_; i++) {
                int64_t stride0 = outputs_[i].byte_strides_[dim0];
                int64_t stride1 = outputs_[i].byte_strides_[dim1];
                if (stride0 == 0 && stride1 != 0) {
                    return -1;
                } else if (stride1 == 0 && stride0 != 0) {
                    return 1;
                } else if (stride0 != 0 && stride1 != 0) {
                    if (stride0 <= stride1) {
                        return -1;
                    } else {
                        return 1;
                    }
                }
            }

            // Inputs
            for (int64_t i = 0; i < num_inputs_; i++) {
                int64_t stride0 = inputs_[i].byte_strides_[dim0];
                int64_t stride1 = inputs_[i].byte_strides_[dim1];
                if (stride0 == 0 || stride1 == 0) {
                    continue;
                } else if (stride0 <= stride1) {
                    return -1;
                } else {
                    return 1;
                }
            }

            return 0;
        };

        // Insertion sort with support for ambiguous comparisons
        for (int i = 1; i < ndims_; i++) {
            int dim1 = i;
            for (int dim0 = i - 1; dim0 >= 0; dim0--) {
                int comparison = ShouldSwap(permute[dim0], permute[dim1]);
                if (comparison > 0) {
                    std::swap(permute[dim0], permute[dim1]);
                    dim1 = dim0;
                } else if (comparison < 0) {
                    break;
                }
            }
        }

        for (int64_t i = 0; i < num_inputs_; i++) {
            inputs_[i].Permute(permute);
        }
        for (int64_t i = 0; i < num_outputs_; i++) {
            outputs_[i].Permute(permute);
        }
    }



} // namespace core
}// namespace pixelflow