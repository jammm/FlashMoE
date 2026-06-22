//
// Lightweight HIP tensor wrapper to replace CuTe tensor types for FlashMoE HIP port
//

#ifndef FLASHMOE_HIP_TENSOR_HPP
#define FLASHMOE_HIP_TENSOR_HPP

#include <hip/hip_runtime.h>
#include <cstdint>
#include <type_traits>

namespace hip_tensor {

// ============================================================================
// Compile-time integer wrapper (replaces cute::Int<N>)
// ============================================================================
template<int N>
struct Int {
    static constexpr int value = N;
    __host__ __device__ constexpr operator int() const { return N; }
    __host__ __device__ constexpr int operator()() const { return N; }
};

// Special compile-time constants (replaces cute::_0, cute::_1, etc.)
using _0 = Int<0>;
using _1 = Int<1>;
using _2 = Int<2>;
using _3 = Int<3>;
using _4 = Int<4>;

// ============================================================================
// Tuple-like structures for Shape and Stride
// ============================================================================

// Forward declarations
template<typename... Ts>
struct Tuple;

template<>
struct Tuple<> {
    static constexpr int rank = 0;
};

template<typename T0>
struct Tuple<T0> {
    T0 v0;
    static constexpr int rank = 1;
    static constexpr int rank_v = 1;

    __host__ __device__ constexpr Tuple() : v0{} {}
    __host__ __device__ constexpr Tuple(T0 a0) : v0(a0) {}

    template<int I>
    __host__ __device__ constexpr auto get() const {
        static_assert(I == 0, "Index out of range");
        return v0;
    }
};

template<typename T0, typename T1>
struct Tuple<T0, T1> {
    T0 v0;
    T1 v1;
    static constexpr int rank = 2;
    static constexpr int rank_v = 2;

    __host__ __device__ constexpr Tuple() : v0{}, v1{} {}
    __host__ __device__ constexpr Tuple(T0 a0, T1 a1) : v0(a0), v1(a1) {}

    template<int I>
    __host__ __device__ constexpr auto get() const {
        if constexpr (I == 0) return v0;
        else if constexpr (I == 1) return v1;
        else static_assert(I < 2, "Index out of range");
    }
};

template<typename T0, typename T1, typename T2>
struct Tuple<T0, T1, T2> {
    T0 v0;
    T1 v1;
    T2 v2;
    static constexpr int rank = 3;
    static constexpr int rank_v = 3;

    __host__ __device__ constexpr Tuple() : v0{}, v1{}, v2{} {}
    __host__ __device__ constexpr Tuple(T0 a0, T1 a1, T2 a2) : v0(a0), v1(a1), v2(a2) {}

    template<int I>
    __host__ __device__ constexpr auto get() const {
        if constexpr (I == 0) return v0;
        else if constexpr (I == 1) return v1;
        else if constexpr (I == 2) return v2;
        else static_assert(I < 3, "Index out of range");
    }
};

template<typename T0, typename T1, typename T2, typename T3>
struct Tuple<T0, T1, T2, T3> {
    T0 v0;
    T1 v1;
    T2 v2;
    T3 v3;
    static constexpr int rank = 4;
    static constexpr int rank_v = 4;

    __host__ __device__ constexpr Tuple() : v0{}, v1{}, v2{}, v3{} {}
    __host__ __device__ constexpr Tuple(T0 a0, T1 a1, T2 a2, T3 a3)
        : v0(a0), v1(a1), v2(a2), v3(a3) {}

    template<int I>
    __host__ __device__ constexpr auto get() const {
        if constexpr (I == 0) return v0;
        else if constexpr (I == 1) return v1;
        else if constexpr (I == 2) return v2;
        else if constexpr (I == 3) return v3;
        else static_assert(I < 4, "Index out of range");
    }
};

// get<I> free function
template<int I, typename... Ts>
__host__ __device__ constexpr auto get(const Tuple<Ts...>& t) {
    return t.template get<I>();
}

// ============================================================================
// Shape type (compile-time or runtime dimensions)
// ============================================================================
template<typename... Dims>
struct Shape : Tuple<Dims...> {
    using Base = Tuple<Dims...>;
    using Base::Base;

    static constexpr int rank_v = sizeof...(Dims);

    __host__ __device__ constexpr int rank() const { return rank_v; }
};

// Shape deduction guide
template<typename... Dims>
Shape(Dims...) -> Shape<Dims...>;

// ============================================================================
// Stride type
// ============================================================================
template<typename... Strides>
struct Stride : Tuple<Strides...> {
    using Base = Tuple<Strides...>;
    using Base::Base;

    static constexpr int rank_v = sizeof...(Strides);

    __host__ __device__ constexpr int rank() const { return rank_v; }
};

// Stride deduction guide
template<typename... Strides>
Stride(Strides...) -> Stride<Strides...>;

// ============================================================================
// Helper to get value from Int<N> or runtime int
// ============================================================================
template<typename T>
__host__ __device__ constexpr int to_int(T val) {
    if constexpr (std::is_integral_v<T>) {
        return static_cast<int>(val);
    } else {
        return static_cast<int>(val);  // Relies on operator int()
    }
}

// ============================================================================
// Layout (combines shape and stride)
// ============================================================================
template<typename ShapeT, typename StrideT>
struct Layout {
    ShapeT shape_;
    StrideT stride_;

    __host__ __device__ constexpr Layout() = default;
    __host__ __device__ constexpr Layout(ShapeT s, StrideT st) : shape_(s), stride_(st) {}

    __host__ __device__ constexpr const ShapeT& shape() const { return shape_; }
    __host__ __device__ constexpr const StrideT& stride() const { return stride_; }

    // Size: product of all dimensions
    __host__ __device__ constexpr int size() const {
        return size_impl(std::make_index_sequence<ShapeT::rank_v>{});
    }

    // Linear index calculation for 1D access
    __host__ __device__ constexpr int operator()(int i) const {
        return i;
    }

    // Linear index calculation for 2D access
    __host__ __device__ constexpr int operator()(int i, int j) const {
        return i * to_int(stride_.template get<0>()) + j * to_int(stride_.template get<1>());
    }

    // Linear index calculation for 3D access
    __host__ __device__ constexpr int operator()(int i, int j, int k) const {
        if constexpr (StrideT::rank_v >= 3) {
            return i * to_int(stride_.template get<0>()) +
                   j * to_int(stride_.template get<1>()) +
                   k * to_int(stride_.template get<2>());
        } else {
            return i * to_int(stride_.template get<0>()) + j * to_int(stride_.template get<1>());
        }
    }

    // Linear index calculation for 4D access
    __host__ __device__ constexpr int operator()(int i, int j, int k, int l) const {
        if constexpr (StrideT::rank_v >= 4) {
            return i * to_int(stride_.template get<0>()) +
                   j * to_int(stride_.template get<1>()) +
                   k * to_int(stride_.template get<2>()) +
                   l * to_int(stride_.template get<3>());
        } else if constexpr (StrideT::rank_v >= 3) {
            return i * to_int(stride_.template get<0>()) +
                   j * to_int(stride_.template get<1>()) +
                   k * to_int(stride_.template get<2>());
        } else {
            return i * to_int(stride_.template get<0>()) + j * to_int(stride_.template get<1>());
        }
    }

private:
    template<size_t... Is>
    __host__ __device__ constexpr int size_impl(std::index_sequence<Is...>) const {
        return (to_int(get<Is>(shape_)) * ... * 1);
    }
};

// ============================================================================
// Tensor class
// ============================================================================
template<typename T, typename LayoutT>
struct Tensor {
    T* ptr_;
    LayoutT layout_;

    __host__ __device__ constexpr Tensor() : ptr_(nullptr), layout_() {}
    __host__ __device__ constexpr Tensor(T* p, LayoutT l) : ptr_(p), layout_(l) {}

    // Data access
    __host__ __device__ T* data() const { return ptr_; }
    __host__ __device__ T*& data() { return ptr_; }

    // Layout access
    __host__ __device__ constexpr const LayoutT& layout() const { return layout_; }
    __host__ __device__ constexpr LayoutT& layout() { return layout_; }

    // Size
    __host__ __device__ constexpr int size() const { return layout_.size(); }

    // Element access - 1D
    __host__ __device__ T& operator()(int i) {
        return ptr_[layout_(i)];
    }
    __host__ __device__ const T& operator()(int i) const {
        return ptr_[layout_(i)];
    }

    // Element access - 2D
    __host__ __device__ T& operator()(int i, int j) {
        return ptr_[layout_(i, j)];
    }
    __host__ __device__ const T& operator()(int i, int j) const {
        return ptr_[layout_(i, j)];
    }

    // Element access - 3D
    __host__ __device__ T& operator()(int i, int j, int k) {
        return ptr_[layout_(i, j, k)];
    }
    __host__ __device__ const T& operator()(int i, int j, int k) const {
        return ptr_[layout_(i, j, k)];
    }

    // Element access - 4D
    __host__ __device__ T& operator()(int i, int j, int k, int l) {
        return ptr_[layout_(i, j, k, l)];
    }
    __host__ __device__ const T& operator()(int i, int j, int k, int l) const {
        return ptr_[layout_(i, j, k, l)];
    }

    // Shape access helpers
    template<int I>
    __host__ __device__ constexpr auto extent() const {
        return get<I>(layout_.shape());
    }
};

// ============================================================================
// Factory functions
// ============================================================================

// make_gmem_ptr / make_smem_ptr: simple passthrough (CuTe compatibility)
template<typename T>
__host__ __device__ constexpr T* make_gmem_ptr(T* ptr) {
    return ptr;
}

template<typename T>
__host__ __device__ constexpr T* make_smem_ptr(T* ptr) {
    return ptr;
}

// make_shape: create a Shape from dimensions
template<typename... Dims>
__host__ __device__ constexpr auto make_shape(Dims... dims) {
    return Shape<Dims...>(dims...);
}

// make_stride: create a Stride from stride values
template<typename... Strides>
__host__ __device__ constexpr auto make_stride(Strides... strides) {
    return Stride<Strides...>(strides...);
}

// LayoutRight tag (row-major layout construction)
struct LayoutRight {};

// make_layout: create a Layout from shape and stride
template<typename ShapeT, typename StrideT,
    typename = std::enable_if_t<!std::is_same_v<std::decay_t<StrideT>, LayoutRight>>>
__host__ __device__ constexpr auto make_layout(ShapeT shape, StrideT stride) {
    return Layout<ShapeT, StrideT>(shape, stride);
}

// make_layout with LayoutRight tag (row-major: stride = (dim1, 1))
template<typename T0, typename T1>
__host__ __device__ constexpr auto make_layout(Tuple<T0, T1> shape, LayoutRight) {
    using StrideType = Stride<T1, Int<1>>;
    return Layout<Tuple<T0, T1>, StrideType>(shape, StrideType(hip_tensor::get<1>(shape), Int<1>{}));
}

// make_tensor: create a Tensor from pointer and layout
template<typename T, typename LayoutT>
__host__ __device__ constexpr auto make_tensor(T* ptr, LayoutT layout) {
    return Tensor<T, LayoutT>(ptr, layout);
}

// ============================================================================
// Coordinate utilities (for CuTe compatibility)
// ============================================================================

template<typename... Ts>
__host__ __device__ constexpr auto make_coord(Ts... vals) {
    return Tuple<Ts...>(vals...);
}

// Underscore placeholder for slicing (like cute::_)
struct Underscore {
    __host__ __device__ constexpr Underscore() = default;
};
static constexpr Underscore _;

// ============================================================================
// AlignedArray (replaces cutlass::AlignedArray)
// ============================================================================
template<typename T, int N, int Align = alignof(T)>
struct alignas(Align) AlignedArray {
    using value_type = T;

    T data_[N];

    __host__ __device__ T& operator[](int i) { return data_[i]; }
    __host__ __device__ const T& operator[](int i) const { return data_[i]; }

    __host__ __device__ T* data() { return data_; }
    __host__ __device__ const T* data() const { return data_; }

    static constexpr int size() { return N; }

    // Fill all elements with a value
    __host__ __device__ void fill(const T& val) {
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            data_[i] = val;
        }
    }

    // Clear to zero
    __host__ __device__ void clear() {
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            data_[i] = T{};
        }
    }
};

// ============================================================================
// Helper utilities
// ============================================================================

// Compile-time min
template<int A, int B>
struct MinVal {
    static constexpr int value = (A < B) ? A : B;
};

template<int A, int B>
constexpr int min_v = MinVal<A, B>::value;

// Runtime min (same type)
template<typename T>
__host__ __device__ constexpr T min(T a, T b) {
    return (a < b) ? a : b;
}

// Runtime min (mixed types)
template<typename T, typename U>
__host__ __device__ constexpr auto min(T a, U b) -> decltype(a + b) {
    return (a < b) ? a : b;
}

// Compile-time max
template<int A, int B>
struct MaxVal {
    static constexpr int value = (A > B) ? A : B;
};

template<int A, int B>
constexpr int max_v = MaxVal<A, B>::value;

// Runtime max (2 args)
template<typename T>
__host__ __device__ constexpr T max(T a, T b) {
    return (a > b) ? a : b;
}

// Variadic max (3+ args)
template<typename T, typename... Rest>
__host__ __device__ constexpr T max(T a, T b, Rest... rest) {
    return max((a > b) ? a : b, rest...);
}

// Runtime ceil_div
template<typename T>
__host__ __device__ constexpr T ceil_div(T a, T b) {
    return (a + b - T(1)) / b;
}

template<typename T, typename U>
__host__ __device__ constexpr auto ceil_div(T a, U b)
    -> std::enable_if_t<!std::is_same_v<T, U>, std::common_type_t<T, U>> {
    using C = std::common_type_t<T, U>;
    return (static_cast<C>(a) + static_cast<C>(b) - C(1)) / static_cast<C>(b);
}

// Power of 2 check
template<int N>
struct is_pow2 {
    static constexpr bool value = (N > 0) && ((N & (N - 1)) == 0);
};

template<int N>
constexpr bool is_pow2_v = is_pow2<N>::value;

// ============================================================================
// Conditional return (for CuTe compatibility patterns)
// ============================================================================
template<bool Cond, typename T, typename F>
__host__ __device__ constexpr auto conditional_return(T t_val, F f_val) {
    if constexpr (Cond) {
        return t_val;
    } else {
        return f_val;
    }
}

// ============================================================================
// C template for arrangement compatibility
// ============================================================================
template<typename Arrangement>
struct C {
    using type = Arrangement;
};

} // namespace hip_tensor

// Alias for CuTe compatibility — allows existing code using cute:: to compile
namespace cute = hip_tensor;

#endif // FLASHMOE_HIP_TENSOR_HPP
