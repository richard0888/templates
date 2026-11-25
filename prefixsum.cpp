// Multidimensional prefix "sums" over any commutative operation, plus
// difference arrays (the inverse).
//
//   PrefixSum<T, D, Op>   static queries on a D-dimensional grid
//     * build:    O(D * N), where N is the number of cells
//     * prefix:   O(D)      fold of the box [0, hi], for any Op
//     * query:    O(2^D)    fold of any box [lo, hi], only if Op has an inverse
//
//   Difference<T, D, Op>  many box "apply v" updates, then read every cell
//     * add:   O(2^D)       (Op needs an inverse)
//     * build: O(D * N)
//
// Choosing the function
// ---------------------
// Op is a small policy object with:
//     T identity() const;                         // neutral element
//     T combine(const T& a, const T& b) const;    // must be associative
//                                                 // and commutative
//     T uncombine(const T& a, const T& b) const;  // OPTIONAL: undoes combine,
//                                                 // so combine(x, b), then
//                                                 // uncombine(that, b) gives x
// With uncombine, (T, combine) is an abelian group, and box queries work
// through inclusion-exclusion. Without it (max, min, gcd, ...), only prefix()
// is available, and calling query() fails at compile time.
//
// Built-in ops: Plus (default), Xor, Max, Min.
// For a custom op, write a struct or use make_op with lambdas:
//     auto mul = prefixsum::make_op(1LL, [](ll a, ll b) { return a * b % M; },
//                                        [](ll a, ll b) { return a * inv(b) % M; });
//     prefixsum::PrefixSum<ll, 2, decltype(mul)> ps(dims, values, mul);
//
// The build does one 1D prefix pass per dimension. After pass d, each cell holds
// the fold over all cells that match it on dimensions > d and are <= it on
// dimensions <= d. After all D passes, cell p holds the fold of the box [0, p].
//
// PrefixSum pads the front of every dimension with identity cells, so lookups
// never need bounds checks.
//
// All coordinates are 0-indexed, and all ranges are inclusive: [lo, hi].
//
// Build: g++ -O2 -std=c++17 prefixsum.cpp -o prefixsum

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

namespace prefixsum {

// ---------------------------------------------------------------------------
// Operations

template <class T>
struct Plus {
    T identity() const { return T{}; }
    T combine(const T& a, const T& b) const { return a + b; }
    T uncombine(const T& a, const T& b) const { return a - b; }
};

template <class T>
struct Xor {
    T identity() const { return T{}; }
    T combine(const T& a, const T& b) const { return a ^ b; }
    T uncombine(const T& a, const T& b) const { return a ^ b; }
};

template <class T>
struct Max {  // prefix() only
    T identity() const { return std::numeric_limits<T>::lowest(); }
    T combine(const T& a, const T& b) const { return std::max(a, b); }
};

template <class T>
struct Min {  // prefix() only
    T identity() const { return std::numeric_limits<T>::max(); }
    T combine(const T& a, const T& b) const { return std::min(a, b); }
};

// Op built from lambdas: make_op(identity, combine) or
// make_op(identity, combine, uncombine).
template <class T, class F>
struct LambdaMonoid {
    T id;
    F f;
    T identity() const { return id; }
    T combine(const T& a, const T& b) const { return f(a, b); }
};

template <class T, class F, class G>
struct LambdaGroup {
    T id;
    F f;
    G g;
    T identity() const { return id; }
    T combine(const T& a, const T& b) const { return f(a, b); }
    T uncombine(const T& a, const T& b) const { return g(a, b); }
};

template <class T, class F>
LambdaMonoid<T, F> make_op(T id, F f) { return {id, f}; }

template <class T, class F, class G>
LambdaGroup<T, F, G> make_op(T id, F f, G g) { return {id, f, g}; }

// True if Op provides uncombine, which makes box queries possible.
template <class Op, class T, class = void>
struct is_invertible : std::false_type {};
template <class Op, class T>
struct is_invertible<Op, T,
                     std::void_t<decltype(std::declval<const Op&>().uncombine(
                         std::declval<const T&>(), std::declval<const T&>()))>>
    : std::true_type {};

namespace detail {

// Runs an in-place inclusive scan along each dimension of a row-major grid.
// sizes[d] is the extent of dimension d, and stride[d] is the flat distance
// between neighbours along d. Each pass walks memory sequentially.
template <class T, std::size_t D, class Op>
void sweep(std::vector<T>& a, const std::array<std::size_t, D>& sizes,
           const std::array<std::size_t, D>& stride, const Op& op) {
    const std::size_t total = a.size();
    for (std::size_t d = 0; d < D; ++d) {
        const std::size_t inner = stride[d];         // cells per slice along d
        const std::size_t block = inner * sizes[d];  // one full run along d
        if (block == 0) return;
        for (std::size_t base = 0; base < total; base += block)
            for (std::size_t c = 1; c < sizes[d]; ++c) {
                T* cur = &a[base + c * inner];
                const T* prev = cur - inner;
                for (std::size_t k = 0; k < inner; ++k) cur[k] = op.combine(prev[k], cur[k]);
            }
    }
}

// Row-major strides: the last dimension is contiguous.
template <std::size_t D>
std::array<std::size_t, D> strides(const std::array<std::size_t, D>& sizes) {
    std::array<std::size_t, D> s{};
    std::size_t acc = 1;
    for (std::size_t d = D; d-- > 0;) {
        s[d] = acc;
        acc *= sizes[d];
    }
    return s;
}

template <std::size_t D>
std::size_t volume(const std::array<std::size_t, D>& sizes) {
    std::size_t v = 1;
    for (std::size_t x : sizes) v *= x;
    return v;
}

}  // namespace detail

// ---------------------------------------------------------------------------

template <class T, std::size_t D, class Op = Plus<T>>
class PrefixSum {
    static_assert(D >= 1, "need at least one dimension");

public:
    using Index = std::array<std::size_t, D>;

    // Builds a grid full of identity values. Fill it with at(), then call build().
    explicit PrefixSum(const Index& dims, Op op = Op{}) : dims_(dims), op_(std::move(op)) {
        for (std::size_t d = 0; d < D; ++d) size_[d] = dims[d] + 1;
        stride_ = detail::strides(size_);
        a_.assign(detail::volume(size_), op_.identity());
    }

    // Builds from a row-major array of prod(dims) values. Ready to query.
    PrefixSum(const Index& dims, const std::vector<T>& values, Op op = Op{})
        : PrefixSum(dims, std::move(op)) {
        assert(values.size() == detail::volume(dims));
        Index p{};
        for (std::size_t i = 0; i < values.size(); ++i) {
            at(p) = values[i];
            for (std::size_t d = D; d-- > 0;) {  // odometer increment
                if (++p[d] < dims[d]) break;
                p[d] = 0;
            }
        }
        build();
    }

    // Raw cell access. Only meaningful before build().
    T& at(const Index& p) {
        assert(!built_);
        return a_[padded(p)];
    }

    void build() {
        assert(!built_);
        detail::sweep(a_, size_, stride_, op_);
        built_ = true;
    }

    // Fold over the box lo[d] <= x[d] <= hi[d]. An empty box returns identity.
    // Needs Op::uncombine.
    T query(const Index& lo, const Index& hi) const {
        static_assert(is_invertible<Op, T>::value,
                      "query(lo, hi) needs Op::uncombine; use prefix(hi) for "
                      "non-invertible ops like max/min");
        assert(built_);
        for (std::size_t d = 0; d < D; ++d) {
            if (lo[d] > hi[d]) return op_.identity();
            assert(hi[d] < dims_[d]);
        }
        // Positive corners first, then negative ones. This keeps intermediate
        // values sensible (e.g. for unsigned T).
        T pos = op_.identity(), neg = op_.identity();
        for (unsigned mask = 0; mask < (1u << D); ++mask) {
            std::size_t idx = 0;
            bool odd = false;
            for (std::size_t d = 0; d < D; ++d) {
                if (mask >> d & 1) {
                    idx += lo[d] * stride_[d];  // padded lo - 1
                    odd = !odd;
                } else {
                    idx += (hi[d] + 1) * stride_[d];
                }
            }
            if (odd) neg = op_.combine(neg, a_[idx]);
            else pos = op_.combine(pos, a_[idx]);
        }
        return op_.uncombine(pos, neg);
    }

    // Fold over the box [0, hi]. Works for any Op.
    T prefix(const Index& hi) const {
        assert(built_);
        return a_[padded(hi)];
    }

    const Index& dims() const { return dims_; }
    const Op& op() const { return op_; }

private:
    std::size_t padded(const Index& p) const {
        std::size_t idx = 0;
        for (std::size_t d = 0; d < D; ++d) {
            assert(p[d] < dims_[d]);
            idx += (p[d] + 1) * stride_[d];
        }
        return idx;
    }

    Index dims_{}, size_{}, stride_{};
    Op op_;
    std::vector<T> a_;
    bool built_ = false;
};

template <class T, std::size_t D, class Op = Plus<T>>
class Difference {
    static_assert(D >= 1, "need at least one dimension");
    static_assert(is_invertible<Op, T>::value, "Difference needs Op::uncombine");

public:
    using Index = std::array<std::size_t, D>;

    explicit Difference(const Index& dims, Op op = Op{}) : dims_(dims), op_(std::move(op)) {
        // One extra slot per dimension takes the "hi + 1" corners.
        for (std::size_t d = 0; d < D; ++d) size_[d] = dims[d] + 1;
        stride_ = detail::strides(size_);
        a_.assign(detail::volume(size_), op_.identity());
    }

    // Combines v into every cell in the box [lo, hi]. Call before build().
    void add(const Index& lo, const Index& hi, const T& v) {
        assert(!built_);
        for (std::size_t d = 0; d < D; ++d) {
            if (lo[d] > hi[d]) return;
            assert(hi[d] < dims_[d]);
        }
        for (unsigned mask = 0; mask < (1u << D); ++mask) {
            std::size_t idx = 0;
            bool odd = false;
            for (std::size_t d = 0; d < D; ++d) {
                if (mask >> d & 1) {
                    idx += (hi[d] + 1) * stride_[d];
                    odd = !odd;
                } else {
                    idx += lo[d] * stride_[d];
                }
            }
            a_[idx] = odd ? op_.uncombine(a_[idx], v) : op_.combine(a_[idx], v);
        }
    }

    void build() {
        assert(!built_);
        detail::sweep(a_, size_, stride_, op_);
        built_ = true;
    }

    // Final value of cell p. Call after build().
    T get(const Index& p) const {
        assert(built_);
        std::size_t idx = 0;
        for (std::size_t d = 0; d < D; ++d) {
            assert(p[d] < dims_[d]);
            idx += p[d] * stride_[d];
        }
        return a_[idx];
    }

    const Index& dims() const { return dims_; }
    const Op& op() const { return op_; }

private:
    Index dims_{}, size_{}, stride_{};
    Op op_;
    std::vector<T> a_;
    bool built_ = false;
};

}  // namespace prefixsum

// ---------------------------------------------------------------------------
// Self-test: compares both structures against brute force on random grids,
// for several operations.

namespace {

std::mt19937_64 rng(12345);

std::size_t rnd(std::size_t lo, std::size_t hi) {  // inclusive
    return std::uniform_int_distribution<std::size_t>(lo, hi)(rng);
}

template <std::size_t D>
bool next_cell(std::array<std::size_t, D>& p, const std::array<std::size_t, D>& dims) {
    for (std::size_t d = D; d-- > 0;) {
        if (++p[d] < dims[d]) return true;
        p[d] = 0;
    }
    return false;
}

template <std::size_t D>
bool inside(const std::array<std::size_t, D>& p, const std::array<std::size_t, D>& lo,
            const std::array<std::size_t, D>& hi) {
    for (std::size_t d = 0; d < D; ++d)
        if (p[d] < lo[d] || p[d] > hi[d]) return false;
    return true;
}

template <std::size_t D>
std::size_t flat(const std::array<std::size_t, D>& p, const std::array<std::size_t, D>& dims) {
    std::size_t i = 0;
    for (std::size_t d = 0; d < D; ++d) i = i * dims[d] + p[d];
    return i;
}

template <std::size_t D>
void random_box(const std::array<std::size_t, D>& dims, std::array<std::size_t, D>& lo,
                std::array<std::size_t, D>& hi) {
    for (std::size_t d = 0; d < D; ++d) {
        lo[d] = rnd(0, dims[d] - 1);
        hi[d] = rnd(0, dims[d] - 1);
        if (rng() % 8 && lo[d] > hi[d]) std::swap(lo[d], hi[d]);  // some empty boxes
    }
}

// gen() makes a random value. Invertible ops test query() and Difference;
// other ops test prefix().
template <std::size_t D, class T, class Op, class Gen>
int test(const char* name, std::size_t max_side, int trials, Op op, Gen gen) {
    using Index = std::array<std::size_t, D>;
    constexpr bool inv = prefixsum::is_invertible<Op, T>::value;
    int failures = 0;
    for (int t = 0; t < trials; ++t) {
        Index dims;
        for (auto& x : dims) x = rnd(1, max_side);
        std::size_t n = 1;
        for (auto x : dims) n *= x;

        std::vector<T> vals(n);
        for (auto& v : vals) v = gen();

        prefixsum::PrefixSum<T, D, Op> ps(dims, vals, op);
        for (int q = 0; q < 50; ++q) {
            Index lo{}, hi;
            random_box(dims, lo, hi);
            if constexpr (!inv) lo = Index{};
            T want = op.identity();
            Index p{};
            do {
                if (inside(p, lo, hi)) want = op.combine(want, vals[flat(p, dims)]);
            } while (next_cell(p, dims));
            T got;
            if constexpr (inv) got = ps.query(lo, hi);
            else got = ps.prefix(hi);
            if (got != want) ++failures;
        }

        if constexpr (inv) {
            prefixsum::Difference<T, D, Op> df(dims, op);
            std::vector<T> brute(n, op.identity());
            for (int u = 0; u < 30; ++u) {
                Index lo, hi;
                random_box(dims, lo, hi);
                T v = gen();
                df.add(lo, hi, v);
                Index p{};
                do {
                    if (inside(p, lo, hi)) {
                        T& b = brute[flat(p, dims)];
                        b = op.combine(b, v);
                    }
                } while (next_cell(p, dims));
            }
            df.build();
            Index p{};
            do {
                if (df.get(p) != brute[flat(p, dims)]) ++failures;
            } while (next_cell(p, dims));
        }
    }
    std::printf("%-6s D=%zu: %s\n", name, D, failures ? "FAIL" : "ok");
    return failures;
}

}  // namespace

int main() {
    using ll = long long;
    using ull = unsigned long long;

    // Example: 2D grid, sum of the 2x2 box at rows 1..2, cols 1..2.
    prefixsum::PrefixSum<int, 2> g({3, 4}, {1, 2, 3, 4,
                                            5, 6, 7, 8,
                                            9, 10, 11, 12});
    std::printf("sum example: %d (expected 34)\n", g.query({1, 1}, {2, 2}));

    // Same grid, max over the box [0, (1, 2)].
    prefixsum::PrefixSum<int, 2, prefixsum::Max<int>> gm({3, 4}, {1, 2, 3, 4,
                                                                  5, 6, 7, 8,
                                                                  9, 10, 11, 12});
    std::printf("max example: %d (expected 7)\n", gm.prefix({1, 2}));

    auto small = [] { return (ll)rnd(0, 2000) - 1000; };
    auto bits = [] { return (ull)rng(); };

    // Custom op from lambdas: addition mod a prime.
    const ll M = 1'000'000'007;
    auto addmod = prefixsum::make_op(
        0LL, [M](ll a, ll b) { return (a + b) % M; },
        [M](ll a, ll b) { return (a - b + M) % M; });
    auto modval = [M] { return (ll)rnd(0, M - 1); };

    // Custom op without an inverse: gcd (prefix only).
    auto gcd = prefixsum::make_op(0LL, [](ll a, ll b) {
        while (b) { ll t = a % b; a = b; b = t; }
        return a;
    });
    auto gval = [] { return (ll)rnd(1, 60); };

    int bad = 0;
#define RUN(D, side, trials)                                                              \
    bad += test<D, ll>("plus", side, trials, prefixsum::Plus<ll>{}, small);              \
    bad += test<D, ull>("xor", side, trials, prefixsum::Xor<ull>{}, bits);               \
    bad += test<D, ll>("max", side, trials, prefixsum::Max<ll>{}, small);                \
    bad += test<D, ll>("min", side, trials, prefixsum::Min<ll>{}, small);                \
    bad += test<D, ll>("addmod", side, trials, addmod, modval);                          \
    bad += test<D, ll>("gcd", side, trials, gcd, gval);
    RUN(1, 40, 100)
    RUN(2, 15, 100)
    RUN(3, 8, 100)
    RUN(4, 5, 60)
    RUN(5, 4, 30)
#undef RUN
    return bad ? 1 : 0;
}
