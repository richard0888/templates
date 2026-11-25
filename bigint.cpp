// BigInt — arbitrary-precision signed integer.
//
//   * +, -, comparisons                         O(n)
//   * *                                         O(n^1.585)  (Karatsuba above a threshold)
//   * /, %  (truncating, like built-in ints)    O(n * m)    (Knuth algorithm D)
//   * parse / to_string                         O(n)
//
// Magnitude is stored little-endian in base 1e9 limbs (uint32_t), so decimal
// I/O is linear and each limb product fits comfortably in a uint64_t.
// Sign is a separate flag; zero is always non-negative with an empty limb vector.
//
// Semantics match C++ built-in integers: a / b truncates toward zero and
// a % b has the sign of a, so (a / b) * b + a % b == a. Division by zero throws.
//
// Build: g++ -O2 -std=c++17 bigint.cpp -o bigint

#include <algorithm>
#include <cassert>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

class BigInt {
  public:
    using Limb = uint32_t;
    using Mag = std::vector<Limb>;
    static constexpr Limb BASE = 1000000000;
    static constexpr int BASE_DIGITS = 9;

    BigInt() = default;
    BigInt(long long v) {  // implicit on purpose: allows `x + 1`, `x == 0`
        neg_ = v < 0;
        unsigned long long u = neg_ ? 0ULL - (unsigned long long)v : (unsigned long long)v;
        while (u) {
            mag_.push_back(Limb(u % BASE));
            u /= BASE;
        }
    }
    explicit BigInt(const std::string& s) { parse(s); }  // also takes string literals

    // ---- observers ----------------------------------------------------------
    bool is_zero() const { return mag_.empty(); }
    bool is_negative() const { return neg_; }
    int sign() const { return is_zero() ? 0 : (neg_ ? -1 : 1); }
    explicit operator bool() const { return !is_zero(); }

    std::string to_string() const {
        if (is_zero()) return "0";
        std::string s = neg_ ? "-" : "";
        s += std::to_string(mag_.back());
        char buf[16];
        for (size_t i = mag_.size() - 1; i-- > 0;) {
            std::snprintf(buf, sizeof buf, "%09u", (unsigned)mag_[i]);
            s += buf;
        }
        return s;
    }

    // Throws std::overflow_error if the value doesn't fit.
    long long to_ll() const {
        unsigned long long u = 0, limit = neg_ ? 9223372036854775808ULL : 9223372036854775807ULL;
        for (size_t i = mag_.size(); i-- > 0;) {
            if (u > (limit - mag_[i]) / BASE) throw std::overflow_error("BigInt::to_ll");
            u = u * BASE + mag_[i];
        }
        return neg_ ? (long long)(0ULL - u) : (long long)u;
    }

    // ---- arithmetic ---------------------------------------------------------
    BigInt operator-() const {
        BigInt r = *this;
        if (!r.is_zero()) r.neg_ = !r.neg_;
        return r;
    }
    BigInt operator+() const { return *this; }

    BigInt& operator+=(const BigInt& o) {
        if (neg_ == o.neg_) {
            add_to(mag_, o.mag_);
        } else if (cmp_mag(mag_, o.mag_) >= 0) {
            sub_from(mag_, o.mag_);
        } else {
            Mag t = o.mag_;
            sub_from(t, mag_);
            mag_ = std::move(t);
            neg_ = o.neg_;
        }
        fix_zero();
        return *this;
    }
    BigInt& operator-=(const BigInt& o) { return *this += -o; }

    BigInt& operator*=(const BigInt& o) {
        mag_ = mul(mag_, o.mag_);
        neg_ = neg_ != o.neg_;
        fix_zero();
        return *this;
    }
    BigInt& operator/=(const BigInt& o) { return *this = divmod(*this, o).first; }
    BigInt& operator%=(const BigInt& o) { return *this = divmod(*this, o).second; }

    BigInt& operator++() { return *this += 1; }
    BigInt& operator--() { return *this -= 1; }
    BigInt operator++(int) { BigInt t = *this; ++*this; return t; }
    BigInt operator--(int) { BigInt t = *this; --*this; return t; }

    friend BigInt operator+(BigInt a, const BigInt& b) { return a += b; }
    friend BigInt operator-(BigInt a, const BigInt& b) { return a -= b; }
    friend BigInt operator*(const BigInt& a, const BigInt& b) { BigInt r = a; return r *= b; }
    friend BigInt operator/(const BigInt& a, const BigInt& b) { return divmod(a, b).first; }
    friend BigInt operator%(const BigInt& a, const BigInt& b) { return divmod(a, b).second; }

    // Truncating division: quotient rounds toward zero, remainder has a's sign.
    static std::pair<BigInt, BigInt> divmod(const BigInt& a, const BigInt& b) {
        if (b.is_zero()) throw std::domain_error("BigInt: division by zero");
        BigInt q, r;
        divmod_mag(a.mag_, b.mag_, q.mag_, r.mag_);
        q.neg_ = a.neg_ != b.neg_;
        r.neg_ = a.neg_;
        q.fix_zero();
        r.fix_zero();
        return {std::move(q), std::move(r)};
    }

    // ---- comparison ---------------------------------------------------------
    friend int compare(const BigInt& a, const BigInt& b) {
        if (a.neg_ != b.neg_) return a.neg_ ? -1 : 1;
        int c = cmp_mag(a.mag_, b.mag_);
        return a.neg_ ? -c : c;
    }
    friend bool operator==(const BigInt& a, const BigInt& b) { return a.neg_ == b.neg_ && a.mag_ == b.mag_; }
    friend bool operator!=(const BigInt& a, const BigInt& b) { return !(a == b); }
    friend bool operator<(const BigInt& a, const BigInt& b) { return compare(a, b) < 0; }
    friend bool operator>(const BigInt& a, const BigInt& b) { return compare(a, b) > 0; }
    friend bool operator<=(const BigInt& a, const BigInt& b) { return compare(a, b) <= 0; }
    friend bool operator>=(const BigInt& a, const BigInt& b) { return compare(a, b) >= 0; }

    // ---- utilities ----------------------------------------------------------
    friend BigInt abs(BigInt a) { a.neg_ = false; return a; }

    friend BigInt pow(BigInt base, unsigned long long e) {
        BigInt r = 1;
        while (e) {
            if (e & 1) r *= base;
            e >>= 1;
            if (e) base *= base;
        }
        return r;
    }

    friend BigInt gcd(BigInt a, BigInt b) {
        a.neg_ = b.neg_ = false;
        while (!b.is_zero()) {
            BigInt t = a % b;
            a = std::move(b);
            b = std::move(t);
        }
        return a;
    }

    friend std::ostream& operator<<(std::ostream& os, const BigInt& x) { return os << x.to_string(); }
    friend std::istream& operator>>(std::istream& is, BigInt& x) {
        std::string s;
        if (is >> s) {
            try { x.parse(s); } catch (const std::invalid_argument&) { is.setstate(std::ios::failbit); }
        }
        return is;
    }

  private:
    Mag mag_;          // little-endian base-1e9 limbs, no leading zero limbs
    bool neg_ = false;

    static constexpr size_t KARATSUBA_THRESHOLD = 48;

    void fix_zero() {
        trim(mag_);
        if (mag_.empty()) neg_ = false;
    }

    // Accepts an optional '+'/'-' followed by one or more decimal digits.
    void parse(const std::string& s) {
        size_t i = 0;
        bool neg = false;
        if (i < s.size() && (s[i] == '+' || s[i] == '-')) neg = s[i++] == '-';
        if (i == s.size()) throw std::invalid_argument("BigInt: empty number");
        for (size_t k = i; k < s.size(); ++k)
            if (s[k] < '0' || s[k] > '9') throw std::invalid_argument("BigInt: bad digit");
        Mag m;
        for (size_t end = s.size(); end > i;) {
            size_t start = end >= i + BASE_DIGITS ? end - BASE_DIGITS : i;
            Limb v = 0;
            for (size_t k = start; k < end; ++k) v = v * 10 + Limb(s[k] - '0');
            m.push_back(v);
            end = start;
        }
        mag_ = std::move(m);
        neg_ = neg;
        fix_zero();
    }

    // ---- magnitude primitives ----------------------------------------------
    static void trim(Mag& a) {
        while (!a.empty() && a.back() == 0) a.pop_back();
    }

    static int cmp_mag(const Mag& a, const Mag& b) {
        if (a.size() != b.size()) return a.size() < b.size() ? -1 : 1;
        for (size_t i = a.size(); i-- > 0;)
            if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
        return 0;
    }

    // a += b
    static void add_to(Mag& a, const Mag& b) {
        if (a.size() < b.size()) a.resize(b.size(), 0);
        Limb carry = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            if (i >= b.size() && !carry) break;
            Limb s = a[i] + carry + (i < b.size() ? b[i] : 0);  // < 2^32
            carry = s >= BASE;
            a[i] = carry ? s - BASE : s;
        }
        if (carry) a.push_back(1);
    }

    // a -= b, requires a >= b
    static void sub_from(Mag& a, const Mag& b) {
        int64_t borrow = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            if (i >= b.size() && !borrow) break;
            int64_t d = int64_t(a[i]) - borrow - (i < b.size() ? b[i] : 0);
            borrow = d < 0;
            a[i] = Limb(borrow ? d + BASE : d);
        }
        assert(!borrow);
        trim(a);
    }

    static Mag mul_school(const Mag& a, const Mag& b) {
        if (a.empty() || b.empty()) return {};
        Mag r(a.size() + b.size(), 0);
        for (size_t i = 0; i < a.size(); ++i) {
            uint64_t carry = 0, ai = a[i];
            if (!ai) continue;
            for (size_t j = 0; j < b.size(); ++j) {
                uint64_t cur = r[i + j] + ai * b[j] + carry;
                r[i + j] = Limb(cur % BASE);
                carry = cur / BASE;
            }
            for (size_t k = i + b.size(); carry; ++k) {
                uint64_t cur = r[k] + carry;
                r[k] = Limb(cur % BASE);
                carry = cur / BASE;
            }
        }
        trim(r);
        return r;
    }

    // r += x * BASE^shift
    static void add_shifted(Mag& r, const Mag& x, size_t shift) {
        if (r.size() < x.size() + shift) r.resize(x.size() + shift, 0);
        Limb carry = 0;
        size_t i = 0;
        for (; i < x.size() || carry; ++i) {
            if (shift + i == r.size()) r.push_back(0);
            Limb s = r[shift + i] + carry + (i < x.size() ? x[i] : 0);
            carry = s >= BASE;
            r[shift + i] = carry ? s - BASE : s;
        }
    }

    static Mag mul(const Mag& a, const Mag& b) {
        if (std::min(a.size(), b.size()) < KARATSUBA_THRESHOLD) return mul_school(a, b);
        size_t k = std::max(a.size(), b.size()) / 2;
        auto lo = [k](const Mag& x) { Mag r(x.begin(), x.begin() + std::min(k, x.size())); trim(r); return r; };
        auto hi = [k](const Mag& x) { return x.size() > k ? Mag(x.begin() + k, x.end()) : Mag(); };
        Mag a0 = lo(a), a1 = hi(a), b0 = lo(b), b1 = hi(b);
        Mag z0 = mul(a0, b0), z2 = mul(a1, b1);
        add_to(a0, a1);
        add_to(b0, b1);
        Mag z1 = mul(a0, b0);  // (a0+a1)(b0+b1) - z0 - z2 = a0*b1 + a1*b0 >= 0
        trim(z1);
        sub_from(z1, z0);
        sub_from(z1, z2);
        Mag r = std::move(z0);
        add_shifted(r, z1, k);
        add_shifted(r, z2, 2 * k);
        trim(r);
        return r;
    }

    // q = a / d, returns a % d   (d < BASE, d != 0)
    static Limb divmod_small(const Mag& a, Limb d, Mag& q) {
        q.assign(a.size(), 0);
        uint64_t rem = 0;
        for (size_t i = a.size(); i-- > 0;) {
            uint64_t cur = rem * BASE + a[i];
            q[i] = Limb(cur / d);
            rem = cur % d;
        }
        trim(q);
        return Limb(rem);
    }

    static Mag mul_small(const Mag& a, Limb m) {
        Mag r(a.size());
        uint64_t carry = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            uint64_t cur = uint64_t(a[i]) * m + carry;
            r[i] = Limb(cur % BASE);
            carry = cur / BASE;
        }
        if (carry) r.push_back(Limb(carry));
        return r;
    }

    // Knuth, TAOCP vol. 2, 4.3.1, Algorithm D (base 1e9).
    static void divmod_mag(const Mag& a, const Mag& b, Mag& q, Mag& r) {
        if (cmp_mag(a, b) < 0) { q.clear(); r = a; return; }
        if (b.size() == 1) {
            Limb rem = divmod_small(a, b[0], q);
            r.clear();
            if (rem) r.push_back(rem);
            return;
        }
        // Normalize so the divisor's top limb is >= BASE/2; this bounds the
        // qhat estimate to be at most 2 too large.
        Limb norm = BASE / (b.back() + 1);
        Mag u = mul_small(a, norm), v = mul_small(b, norm);
        u.resize(a.size() + 1, 0);
        const size_t n = v.size(), m = u.size() - n;
        const uint64_t vtop = v[n - 1], vnext = v[n - 2];
        q.assign(m, 0);

        for (size_t j = m; j-- > 0;) {
            uint64_t num = uint64_t(u[j + n]) * BASE + u[j + n - 1];
            uint64_t qhat = num / vtop, rhat = num % vtop;
            while (qhat >= BASE || qhat * vnext > rhat * BASE + u[j + n - 2]) {
                --qhat;
                rhat += vtop;
                if (rhat >= BASE) break;
            }
            // u[j .. j+n] -= qhat * v
            int64_t borrow = 0;
            uint64_t carry = 0;
            for (size_t i = 0; i < n; ++i) {
                uint64_t p = qhat * v[i] + carry;
                carry = p / BASE;
                int64_t t = int64_t(u[i + j]) - int64_t(p % BASE) - borrow;
                borrow = t < 0;
                u[i + j] = Limb(borrow ? t + BASE : t);
            }
            int64_t top = int64_t(u[j + n]) - int64_t(carry) - borrow;
            if (top < 0) {  // qhat was one too large: add v back (rare)
                --qhat;
                Limb c = 0;
                for (size_t i = 0; i < n; ++i) {
                    Limb s = u[i + j] + v[i] + c;
                    c = s >= BASE;
                    u[i + j] = c ? s - BASE : s;
                }
                top += c;
            }
            assert(top >= 0 && top < int64_t(BASE));
            u[j + n] = Limb(top);
            q[j] = Limb(qhat);
        }
        trim(q);
        u.resize(n);
        trim(u);
        divmod_small(u, norm, r);  // un-normalize remainder (exact)
    }

    friend struct BigIntTest;
};

// ---- tests / benchmark ------------------------------------------------------

struct BigIntTest {
    static BigInt::Mag school(const BigInt& a, const BigInt& b) { return BigInt::mul_school(a.mag_, b.mag_); }
    static const BigInt::Mag& mag(const BigInt& a) { return a.mag_; }
};

static std::string random_digits(std::mt19937_64& rng, size_t len) {
    std::string s(len, '0');
    for (char& c : s) c = char('0' + rng() % 10);
    if (rng() & 1) s = "-" + s;
    return s;
}

int main() {
    BigInt a("3");
    std::cout << pow(a, 1000) << '\n';
    // std::mt19937_64 rng(12345);

    // // Small values vs. __int128 (exercises every operator and sign combination).
    // for (int it = 0; it < 200000; ++it) {
    //     auto pick = [&]() -> long long {
    //         switch (rng() % 4) {
    //             case 0: return (long long)(rng() % 21) - 10;
    //             case 1: return (long long)(rng() % 2000000001) - 1000000000;
    //             case 2: return (long long)(rng() >> (rng() % 63)) * ((rng() & 1) ? 1 : -1);
    //             default: return (long long)rng();
    //         }
    //     };
    //     long long x = pick(), y = pick();
    //     BigInt a = x, b = y;
    //     __int128 X = x, Y = y;
    //     auto eq = [](const BigInt& v, __int128 w) {
    //         BigInt expect;
    //         bool neg = w < 0;
    //         unsigned __int128 u = neg ? (unsigned __int128)0 - (unsigned __int128)w : (unsigned __int128)w;
    //         const unsigned long long E18 = 1000000000000000000ULL;
    //         expect = BigInt((long long)(u % E18)) +
    //                  BigInt((long long)(u / E18 % E18)) * BigInt((long long)E18) +
    //                  BigInt((long long)(u / E18 / E18)) * BigInt((long long)E18) * BigInt((long long)E18);
    //         if (neg) expect = -expect;
    //         return v == expect;
    //     };
    //     assert(a.to_ll() == x && BigInt(a.to_string()) == a);
    //     assert(eq(a + b, X + Y));
    //     assert(eq(a - b, X - Y));
    //     assert(eq(a * b, X * Y));
    //     if (y != 0) {
    //         assert(eq(a / b, X / Y));
    //         assert(eq(a % b, X % Y));
    //     }
    //     assert((a < b) == (x < y) && (a == b) == (x == y) && (a >= b) == (x >= y));
    // }
    // assert(BigInt(LLONG_MIN).to_ll() == LLONG_MIN);
    // assert(BigInt(LLONG_MIN).to_string() == "-9223372036854775808");

    // // Big values: identities and Karatsuba vs. schoolbook.
    // for (int it = 0; it < 300; ++it) {
    //     BigInt a(random_digits(rng, 1 + rng() % 3000));
    //     BigInt b(random_digits(rng, 1 + rng() % 3000));
    //     BigInt p = a * b;
    //     assert(BigIntTest::mag(p) == BigIntTest::school(a, b));
    //     if (!a.is_zero()) assert(p / a == b && p % a == 0);
    //     if (!b.is_zero()) {
    //         auto [q, r] = BigInt::divmod(a, b);
    //         assert(q * b + r == a);
    //         assert(abs(r) < abs(b));
    //         assert(r.is_zero() || r.is_negative() == a.is_negative());
    //     }
    //     assert(a + b - b == a && (a - b) + b == a);
    //     assert(BigInt(a.to_string()) == a);
    // }

    // // Division edge cases that force the qhat correction / add-back paths.
    // {
    //     BigInt big = pow(BigInt(10), 450) - 1;           // all 999999999 limbs
    //     BigInt d = pow(BigInt(10), 180) + 1;
    //     auto [q, r] = BigInt::divmod(big, d);
    //     assert(q * d + r == big && r < d);
    //     BigInt x = pow(BigInt(2), 1000), y = pow(BigInt(2), 500) - 1;
    //     assert((x / y) * y + x % y == x);
    // }

    // // Known values.
    // {
    //     BigInt f = 1;
    //     for (int i = 2; i <= 100; ++i) f *= i;
    //     assert(f.to_string() ==
    //            "93326215443944152681699238856266700490715968264381621468592963895217599993229915608941463976156518286253697920827223758251185210916864000000000000000000000000");
    //     assert(pow(BigInt(2), 128).to_string() == "340282366920938463463374607431768211456");
    //     assert(gcd(BigInt("-123456789012345678901234567890"), BigInt("987654321098765432109876543210")) ==
    //            BigInt("9000000000900000000090"));
    //     assert(BigInt("-0").to_string() == "0" && BigInt("+000123") == 123);
    // }

    // // Error handling.
    // {
    //     bool threw = false;
    //     try { BigInt(1) / BigInt(0); } catch (const std::domain_error&) { threw = true; }
    //     assert(threw);
    //     threw = false;
    //     try { BigInt("12a"); } catch (const std::invalid_argument&) { threw = true; }
    //     assert(threw);
    //     threw = false;
    //     try { pow(BigInt(10), 19).to_ll(); } catch (const std::overflow_error&) { threw = true; }
    //     assert(threw);
    // }

    // std::puts("correctness: OK");

    // // Benchmark.
    // using clk = std::chrono::steady_clock;
    // auto ms = [](clk::time_point s) { return std::chrono::duration<double, std::milli>(clk::now() - s).count(); };
    // BigInt a(random_digits(rng, 100000)), b(random_digits(rng, 100000));
    // auto t = clk::now();
    // BigInt p = a * b;
    // std::printf("mul 100k x 100k digits: %.1f ms\n", ms(t));
    // t = clk::now();
    // BigInt q = p / BigInt(random_digits(rng, 50000));
    // std::printf("div 200k / 50k digits:  %.1f ms\n", ms(t));
    // t = clk::now();
    // std::string s = p.to_string();
    // std::printf("to_string 200k digits:  %.1f ms\n", ms(t));
    // t = clk::now();
    // BigInt f = 1;
    // for (int i = 2; i <= 5000; ++i) f *= i;
    // std::printf("5000! (%zu digits):     %.1f ms\n", f.to_string().size(), ms(t));
    // (void)q;
}
