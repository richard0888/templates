// Block sort (block merge sort) — a stable, in-place merge sort.
//
//   * O(n log n) time, O(1) extra memory (plus an optional fixed-size cache)
//   * Stable: equal elements keep their original relative order
//
// The algorithm (after Kim & Kutzner / WikiSort):
//   1. Insertion-sort small runs of 16-31 elements.
//   2. Merge runs bottom-up, level by level. To merge two sorted runs A and B:
//      a. Pull two internal buffers of ~sqrt(n) *unique* values out of the
//         array. buffer1 tags A blocks; buffer2 serves as swap space for merges.
//      b. Split A into blocks of size ~sqrt(n). Tag each block by swapping its
//         first value with a buffer1 value.
//      c. "Roll" the A blocks through B. Whenever the smallest A block (found by
//         its tag) belongs before the next B block, drop it into place and
//         locally merge the previous A block with the B values that follow it.
//      d. Once the level is done, sort buffer2 and put both buffers back where
//         they came from.
//   If there aren't enough unique values for full buffers, it falls back to
//   rotation-based in-place merging. That fallback is still fast, because few
//   unique values means rotations move little data.
//
// Build: g++ -O2 -std=c++17 blocksort.cpp -o blocksort

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <functional>
#include <iterator>
#include <random>
#include <utility>
#include <vector>

namespace blocksort {
namespace detail {

struct Range {
    std::size_t start = 0, end = 0;
    Range() = default;
    Range(std::size_t s, std::size_t e) : start(s), end(e) {}
    std::size_t length() const { return end - start; }
};

// Walks the array in evenly sized ranges, doubling the range size each level.
// Sizes at one level differ by at most one, so uneven array lengths work too.
class LevelIterator {
public:
    LevelIterator(std::size_t size, std::size_t min_level) : size_(size) {
        std::size_t pow2 = 1;
        while (pow2 * 2 <= size) pow2 *= 2;
        denominator_ = pow2 / min_level;
        decimal_step_ = size / denominator_;
        numerator_step_ = size % denominator_;
        begin();
    }
    void begin() { numerator_ = decimal_ = 0; }
    bool finished() const { return decimal_ >= size_; }
    std::size_t length() const { return decimal_step_; }

    Range nextRange() {
        std::size_t start = decimal_;
        decimal_ += decimal_step_;
        numerator_ += numerator_step_;
        if (numerator_ >= denominator_) {
            numerator_ -= denominator_;
            ++decimal_;
        }
        return Range(start, decimal_);
    }

    bool nextLevel() {
        decimal_step_ += decimal_step_;
        numerator_step_ += numerator_step_;
        if (numerator_step_ >= denominator_) {
            numerator_step_ -= denominator_;
            ++decimal_step_;
        }
        return decimal_step_ < size_;
    }

private:
    std::size_t size_, denominator_, decimal_step_, numerator_step_;
    std::size_t numerator_ = 0, decimal_ = 0;
};

// Records where a run of unique values was pulled from and to, so it can be
// redistributed afterwards.
struct Pull {
    Range range;
    std::size_t count = 0, from = 0, to = 0;
};

template <class It, class Cmp, std::size_t CacheSize>
class Sorter {
    using T = typename std::iterator_traits<It>::value_type;
    static constexpr std::size_t kMinRun = 16;

public:
    Sorter(It first, Cmp comp) : a_(first), comp_(comp) { cache_.reserve(CacheSize); }

    void sort(std::size_t size) {
        if (size < kMinRun) {
            insertionSort(Range(0, size));
            return;
        }
        LevelIterator it(size, kMinRun);
        while (!it.finished()) insertionSort(it.nextRange());
        if (size < 2 * kMinRun) return;

        while (true) {
            if (it.length() < CacheSize)
                mergeLevelWithCache(it);
            else
                mergeLevelInPlace(it);
            if (!it.nextLevel()) break;
        }
    }

private:
    It a_;
    Cmp comp_;
    std::vector<T> cache_;  // never grows beyond CacheSize elements

    bool less(std::size_t i, std::size_t j) { return comp_(a_[i], a_[j]); }

    // ---- primitives -------------------------------------------------------

    void insertionSort(Range r) {
        for (std::size_t i = r.start + 1; i < r.end; ++i) {
            T tmp = std::move(a_[i]);
            std::size_t j = i;
            for (; j > r.start && comp_(tmp, a_[j - 1]); --j) a_[j] = std::move(a_[j - 1]);
            a_[j] = std::move(tmp);
        }
    }

    // Rotate range left by `amount`.
    void rotate(std::size_t amount, Range r) {
        if (r.length() == 0) return;
        std::rotate(a_ + r.start, a_ + r.start + amount, a_ + r.end);
    }

    void blockSwap(std::size_t s1, std::size_t s2, std::size_t count) {
        std::swap_ranges(a_ + s1, a_ + s1 + count, a_ + s2);
    }

    // First index in r whose value is not less than `value` (lower bound).
    std::size_t binaryFirst(const T& value, Range r) {
        return std::lower_bound(a_ + r.start, a_ + r.end, value, comp_) - a_;
    }
    // First index in r whose value is greater than `value` (upper bound).
    std::size_t binaryLast(const T& value, Range r) {
        return std::upper_bound(a_ + r.start, a_ + r.end, value, comp_) - a_;
    }

    // Galloping variants. With `unique` distinct values expected in r, they
    // jump ahead in steps of r.length()/unique, then finish with a binary search.
    std::size_t findFirstForward(const T& value, Range r, std::size_t unique) {
        if (r.length() == 0) return r.start;
        std::size_t skip = std::max<std::size_t>(r.length() / unique, 1), i;
        for (i = r.start + skip; comp_(a_[i - 1], value); i += skip)
            if (i >= r.end - skip) return binaryFirst(value, Range(i, r.end));
        return binaryFirst(value, Range(i - skip, i));
    }
    std::size_t findLastForward(const T& value, Range r, std::size_t unique) {
        if (r.length() == 0) return r.start;
        std::size_t skip = std::max<std::size_t>(r.length() / unique, 1), i;
        for (i = r.start + skip; !comp_(value, a_[i - 1]); i += skip)
            if (i >= r.end - skip) return binaryLast(value, Range(i, r.end));
        return binaryLast(value, Range(i - skip, i));
    }
    std::size_t findFirstBackward(const T& value, Range r, std::size_t unique) {
        if (r.length() == 0) return r.start;
        std::size_t skip = std::max<std::size_t>(r.length() / unique, 1), i;
        for (i = r.end - skip; i > r.start && !comp_(a_[i - 1], value); i -= skip)
            if (i < r.start + skip) return binaryFirst(value, Range(r.start, i));
        return binaryFirst(value, Range(i, i + skip));
    }
    std::size_t findLastBackward(const T& value, Range r, std::size_t unique) {
        if (r.length() == 0) return r.start;
        std::size_t skip = std::max<std::size_t>(r.length() / unique, 1), i;
        for (i = r.end - skip; i > r.start && comp_(value, a_[i - 1]); i -= skip)
            if (i < r.start + skip) return binaryLast(value, Range(r.start, i));
        return binaryLast(value, Range(i, i + skip));
    }

    // ---- merges -----------------------------------------------------------

    void copyToCache(Range r) {
        cache_.assign(std::make_move_iterator(a_ + r.start), std::make_move_iterator(a_ + r.end));
    }

    // A's contents are in the cache; merge them with B into A.start onward.
    void mergeExternal(Range A, Range B) {
        std::size_t ai = 0, bi = B.start, out = A.start;
        const std::size_t aLen = A.length();
        if (B.length() > 0 && aLen > 0) {
            while (true) {
                if (!comp_(a_[bi], cache_[ai])) {
                    a_[out++] = std::move(cache_[ai++]);
                    if (ai == aLen) break;
                } else {
                    a_[out++] = std::move(a_[bi++]);
                    if (bi == B.end) break;
                }
            }
        }
        std::move(cache_.begin() + ai, cache_.begin() + aLen, a_ + out);
    }

    // A's contents were swapped into `buffer`; merge them with B by swapping,
    // so the buffer's own values end up back in the buffer (in some order).
    void mergeInternal(Range A, Range B, Range buffer) {
        std::size_t ac = 0, bc = 0, ins = 0;
        if (B.length() > 0 && A.length() > 0) {
            while (true) {
                if (!comp_(a_[B.start + bc], a_[buffer.start + ac])) {
                    std::swap(a_[A.start + ins], a_[buffer.start + ac]);
                    ++ac; ++ins;
                    if (ac >= A.length()) break;
                } else {
                    std::swap(a_[A.start + ins], a_[B.start + bc]);
                    ++bc; ++ins;
                    if (bc >= B.length()) break;
                }
            }
        }
        blockSwap(buffer.start + ac, A.start + ins, A.length() - ac);
    }

    // Merge with no extra space using binary searches and rotations.
    void mergeInPlace(Range A, Range B) {
        if (A.length() == 0 || B.length() == 0) return;
        while (true) {
            // Move B's values that are smaller than A's first value in front of A.
            std::size_t mid = binaryFirst(a_[A.start], B);
            std::size_t amount = mid - A.end;
            rotate(A.length(), Range(A.start, mid));
            if (B.end == mid) break;
            B.start = mid;
            A = Range(A.start + amount, B.start);
            // Skip A values that are already in place.
            A.start = binaryLast(a_[A.start], A);
            if (A.length() == 0) break;
        }
    }

    // ---- levels -----------------------------------------------------------

    // Runs are short enough that A fits in the cache: ordinary merge.
    void mergeLevelWithCache(LevelIterator& it) {
        it.begin();
        while (!it.finished()) {
            Range A = it.nextRange(), B = it.nextRange();
            if (less(B.end - 1, A.start)) {
                rotate(A.length(), Range(A.start, B.end));  // reversed: swap A and B
            } else if (less(B.start, A.end - 1)) {         // overlapping: merge
                copyToCache(A);
                mergeExternal(A, B);
            }
        }
    }

    void mergeLevelInPlace(LevelIterator& it) {
        const std::size_t len = it.length();
        std::size_t blockSize = static_cast<std::size_t>(std::sqrt(static_cast<double>(len)));
        std::size_t bufferSize = len / blockSize + 1;

        // --- 1. find unique values for buffer1 (tags) and buffer2 (swap space)
        Range buffer1, buffer2;
        Pull pull[2];
        std::size_t pullIndex = 0;
        std::size_t find = bufferSize + bufferSize;
        bool findSeparately = false;

        if (blockSize <= CacheSize) {
            find = bufferSize;  // the cache replaces buffer2
        } else if (find > len) {
            find = bufferSize;  // can't fit both in one run; look for them one at a time
            findSeparately = true;
        }

        it.begin();
        while (!it.finished()) {
            Range A = it.nextRange(), B = it.nextRange();
            std::size_t last, count, index;

            // Scan A from the left for unique values.
            for (last = A.start, count = 1; count < find; last = index, ++count) {
                index = findLastForward(a_[last], Range(last + 1, A.end), find - count);
                if (index == A.end) break;
            }
            index = last;

            if (count >= bufferSize) {
                pull[pullIndex] = {Range(A.start, B.end), count, index, A.start};
                pullIndex = 1;
                if (count == bufferSize + bufferSize) {
                    buffer1 = Range(A.start, A.start + bufferSize);
                    buffer2 = Range(A.start + bufferSize, A.start + count);
                    break;
                } else if (find == bufferSize + bufferSize) {
                    buffer1 = Range(A.start, A.start + count);
                    find = bufferSize;
                } else if (blockSize <= CacheSize) {
                    buffer1 = Range(A.start, A.start + count);
                    break;
                } else if (findSeparately) {
                    buffer1 = Range(A.start, A.start + count);
                    findSeparately = false;
                } else {
                    buffer2 = Range(A.start, A.start + count);
                    break;
                }
            } else if (pullIndex == 0 && count > buffer1.length()) {
                buffer1 = Range(A.start, A.start + count);
                pull[0] = {Range(A.start, B.end), count, index, A.start};
            }

            // Scan B from the right for unique values.
            for (last = B.end - 1, count = 1; count < find; last = index - 1, ++count) {
                index = findFirstBackward(a_[last], Range(B.start, last), find - count);
                if (index == B.start) break;
            }
            index = last;

            if (count >= bufferSize) {
                pull[pullIndex] = {Range(A.start, B.end), count, index, B.end};
                pullIndex = 1;
                if (count == bufferSize + bufferSize) {
                    buffer1 = Range(B.end - count, B.end - bufferSize);
                    buffer2 = Range(B.end - bufferSize, B.end);
                    break;
                } else if (find == bufferSize + bufferSize) {
                    buffer1 = Range(B.end - count, B.end);
                    find = bufferSize;
                } else if (blockSize <= CacheSize) {
                    buffer1 = Range(B.end - count, B.end);
                    break;
                } else if (findSeparately) {
                    buffer1 = Range(B.end - count, B.end);
                    findSeparately = false;
                } else {
                    // buffer1 came from this pair's A: keep its later
                    // redistribution away from buffer2.
                    if (pull[0].range.start == A.start) pull[0].range.end -= pull[1].count;
                    buffer2 = Range(B.end - count, B.end);
                    break;
                }
            } else if (pullIndex == 0 && count > buffer1.length()) {
                buffer1 = Range(B.end - count, B.end);
                pull[0] = {Range(A.start, B.end), count, index, B.end};
            }
        }

        // --- 2. pull the unique values to the start of A / end of B
        for (Pull& p : pull) {
            const std::size_t length = p.count;
            if (p.to < p.from) {
                std::size_t index = p.from;
                for (std::size_t count = 1; count < length; ++count) {
                    index = findFirstBackward(a_[index - 1], Range(p.to, p.from - (count - 1)), length - count);
                    Range r(index + 1, p.from + 1);
                    rotate(r.length() - count, r);
                    p.from = index + count;
                }
            } else if (p.to > p.from) {
                std::size_t index = p.from + 1;
                for (std::size_t count = 1; count < length; ++count) {
                    index = findLastForward(a_[index], Range(index, p.to), length - count);
                    Range r(p.from, index - 1);
                    rotate(count, r);
                    p.from = index - 1 - count;
                }
            }
        }

        // Fewer unique values may mean fewer tags, so use bigger blocks.
        bufferSize = buffer1.length();
        blockSize = len / bufferSize + 1;

        // --- 3. merge each A/B pair
        it.begin();
        while (!it.finished()) {
            Range A = it.nextRange(), B = it.nextRange();

            // Leave out the parts that are holding buffers.
            const std::size_t start = A.start;
            bool empty = false;
            for (const Pull& p : pull) {
                if (start != p.range.start) continue;
                if (p.from > p.to) {
                    A.start += p.count;
                    if (A.length() == 0) empty = true;
                } else if (p.from < p.to) {
                    B.end -= p.count;
                    if (B.length() == 0) empty = true;
                }
            }
            if (empty) continue;

            if (less(B.end - 1, A.start)) {
                rotate(A.length(), Range(A.start, B.end));
            } else if (less(B.start, A.end - 1)) {
                mergeBlocks(A, B, buffer1, buffer2, blockSize);
            }
        }

        // --- 4. restore the buffers
        // mergeInternal shuffles buffer2. buffer1's tags came back in order.
        insertionSort(buffer2);

        for (const Pull& p : pull) {
            std::size_t unique = p.count * 2;
            if (p.from > p.to) {
                // Pulled to the left, so redistribute to the right.
                Range buffer(p.range.start, p.range.start + p.count);
                while (buffer.length() > 0) {
                    std::size_t index = findFirstForward(a_[buffer.start], Range(buffer.end, p.range.end), unique);
                    std::size_t amount = index - buffer.end;
                    rotate(buffer.length(), Range(buffer.start, index));
                    buffer.start += amount + 1;
                    buffer.end += amount;
                    unique -= 2;
                }
            } else if (p.from < p.to) {
                // Pulled to the right, so redistribute to the left.
                Range buffer(p.range.end - p.count, p.range.end);
                while (buffer.length() > 0) {
                    std::size_t index = findLastBackward(a_[buffer.end - 1], Range(p.range.start, buffer.start), unique);
                    std::size_t amount = buffer.start - index;
                    rotate(amount, Range(index, buffer.end));
                    buffer.start -= amount;
                    buffer.end -= amount + 1;
                    unique -= 2;
                }
            }
        }
    }

    // Merge A and B: tag the A blocks, roll them through B, and drop each one
    // into place.
    void mergeBlocks(Range A, Range B, Range buffer1, Range buffer2, std::size_t blockSize) {
        Range blockA = A;
        Range firstA(A.start, A.start + blockA.length() % blockSize);

        // Tag each full A block: swap its first value with a buffer1 value.
        for (std::size_t ia = buffer1.start, i = firstA.end; i < blockA.end; ++ia, i += blockSize)
            std::swap(a_[ia], a_[i]);

        Range lastA = firstA;
        Range lastB(0, 0);
        Range blockB(B.start, B.start + std::min(blockSize, B.length()));
        blockA.start += firstA.length();
        std::size_t indexA = buffer1.start;

        // Move the uneven first A block somewhere we can merge it from.
        if (lastA.length() <= CacheSize)
            copyToCache(lastA);
        else if (buffer2.length() > 0)
            blockSwap(lastA.start, buffer2.start, lastA.length());

        if (blockA.length() > 0) {
            while (true) {
                // Drop the smallest A block when it belongs before the next B
                // block, or when no B blocks are left.
                if ((lastB.length() > 0 && !comp_(a_[lastB.end - 1], a_[indexA])) || blockB.length() == 0) {
                    std::size_t bSplit = binaryFirst(a_[indexA], lastB);
                    std::size_t bRemaining = lastB.end - bSplit;

                    // Find the smallest A block by its tag and move it to the front.
                    std::size_t minA = blockA.start;
                    for (std::size_t f = minA + blockSize; f < blockA.end; f += blockSize)
                        if (less(f, minA)) minA = f;
                    blockSwap(blockA.start, minA, blockSize);

                    // Swap its real first value back in from buffer1.
                    std::swap(a_[blockA.start], a_[indexA]);
                    ++indexA;

                    // Merge the previous A block with the B values after it.
                    mergeLocal(lastA, Range(lastA.end, bSplit), buffer2);

                    if (buffer2.length() > 0 || blockSize <= CacheSize) {
                        // Move the new A block out to where it will be merged
                        // from. That frees its slot, so a block swap does the
                        // job of a rotation.
                        if (blockSize <= CacheSize)
                            copyToCache(Range(blockA.start, blockA.start + blockSize));
                        else
                            blockSwap(blockA.start, buffer2.start, blockSize);
                        blockSwap(bSplit, blockA.start + blockSize - bRemaining, bRemaining);
                    } else {
                        rotate(blockA.start - bSplit, Range(bSplit, blockA.start + blockSize));
                    }

                    lastA = Range(blockA.start - bRemaining, blockA.start - bRemaining + blockSize);
                    lastB = Range(lastA.end, lastA.end + bRemaining);

                    blockA.start += blockSize;
                    if (blockA.length() == 0) break;
                } else if (blockB.length() < blockSize) {
                    // Last B block is short: rotate it in front of the A blocks.
                    rotate(blockB.start - blockA.start, Range(blockA.start, blockB.end));
                    lastB = Range(blockA.start, blockA.start + blockB.length());
                    blockA.start += blockB.length();
                    blockA.end += blockB.length();
                    blockB.end = blockB.start;
                } else {
                    // Roll the leftmost A block past the next B block.
                    blockSwap(blockA.start, blockB.start, blockSize);
                    lastB = Range(blockA.start, blockA.start + blockSize);
                    blockA.start += blockSize;
                    blockA.end += blockSize;
                    blockB.start += blockSize;
                    if (blockB.end > B.end - blockSize)
                        blockB.end = B.end;
                    else
                        blockB.end += blockSize;
                }
            }
        }

        // Merge the final A block with what's left of B.
        mergeLocal(lastA, Range(lastA.end, B.end), buffer2);
    }

    // lastA's contents are in the cache, in buffer2, or still in place,
    // depending on its size and which buffers exist.
    void mergeLocal(Range A, Range B, Range buffer2) {
        if (A.length() <= CacheSize)
            mergeExternal(A, B);
        else if (buffer2.length() > 0)
            mergeInternal(A, B, buffer2);
        else
            mergeInPlace(A, B);
    }
};

}  // namespace detail

// Stable in-place sort of [first, last). CacheSize is a fixed scratch size in
// elements; 0 makes the sort fully in-place (O(1) extra memory).
template <std::size_t CacheSize = 512, class It, class Cmp = std::less<>>
void sort(It first, It last, Cmp comp = Cmp()) {
    const std::size_t n = static_cast<std::size_t>(std::distance(first, last));
    if (n < 2) return;
    detail::Sorter<It, Cmp, CacheSize> sorter(first, comp);
    sorter.sort(n);
}

}  // namespace blocksort

// ============================================================================
// Tests and benchmark
// ============================================================================

struct Item {
    int key;
    int order;  // original position, used to check stability
};

template <std::size_t CacheSize>
bool check(std::vector<Item> v) {
    std::vector<Item> expected = v;
    auto byKey = [](const Item& x, const Item& y) { return x.key < y.key; };
    std::stable_sort(expected.begin(), expected.end(), byKey);
    blocksort::sort<CacheSize>(v.begin(), v.end(), byKey);
    for (std::size_t i = 0; i < v.size(); ++i)
        if (v[i].key != expected[i].key || v[i].order != expected[i].order) return false;
    return true;
}

std::vector<Item> makeInput(std::size_t n, int pattern, std::mt19937& rng) {
    std::vector<Item> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        int k = 0;
        switch (pattern) {
            case 0: k = static_cast<int>(rng()); break;                         // random
            case 1: k = static_cast<int>(rng() % 4); break;                     // 4 unique values
            case 2: k = static_cast<int>(rng() % (n / 8 + 1)); break;           // many duplicates
            case 3: k = static_cast<int>(i); break;                             // sorted
            case 4: k = static_cast<int>(n - i); break;                         // reversed
            case 5: k = static_cast<int>(rng() % 100 == 0 ? rng() : i); break;  // nearly sorted
            case 6: k = static_cast<int>(i % 37); break;                        // sawtooth
            case 7: k = 7; break;                                               // all equal
        }
        v[i] = {k, static_cast<int>(i)};
    }
    return v;
}

int main() {
    std::mt19937 rng(12345);
    const char* names[] = {"random", "4 unique", "many dups", "sorted",
                           "reversed", "nearly sorted", "sawtooth", "all equal"};
    int failures = 0, tests = 0;

    // Correctness and stability, with the cache and with no cache at all.
    std::vector<std::size_t> sizes;
    for (std::size_t n = 0; n <= 1200; ++n) sizes.push_back(n);
    for (std::size_t n : {4095, 4096, 4097, 10000, 65537, 100003, 300000}) sizes.push_back(n);

    for (int pattern = 0; pattern < 8; ++pattern) {
        for (std::size_t n : sizes) {
            auto v = makeInput(n, pattern, rng);
            tests += 2;
            if (!check<512>(v)) { ++failures; std::printf("FAIL cache=512 %s n=%zu\n", names[pattern], n); }
            if (!check<0>(v))   { ++failures; std::printf("FAIL cache=0   %s n=%zu\n", names[pattern], n); }
        }
    }
    std::printf("%d/%d correctness+stability tests passed\n\n", tests - failures, tests);

    // Benchmark against std::stable_sort.
    const std::size_t n = 2000000;
    std::printf("Benchmark, n = %zu ints\n", n);
    std::printf("%-14s %12s %14s %16s\n", "pattern", "block (ms)", "block,0 (ms)", "std::stable (ms)");
    for (int pattern = 0; pattern < 8; ++pattern) {
        std::vector<int> base(n);
        auto items = makeInput(n, pattern, rng);
        for (std::size_t i = 0; i < n; ++i) base[i] = items[i].key;

        auto time = [&](auto&& fn) {
            std::vector<int> v = base;
            auto t0 = std::chrono::steady_clock::now();
            fn(v);
            auto t1 = std::chrono::steady_clock::now();
            if (!std::is_sorted(v.begin(), v.end())) std::printf("  (not sorted!)\n");
            return std::chrono::duration<double, std::milli>(t1 - t0).count();
        };
        double tb = time([](std::vector<int>& v) { blocksort::sort(v.begin(), v.end()); });
        double t0 = time([](std::vector<int>& v) { blocksort::sort<0>(v.begin(), v.end()); });
        double ts = time([](std::vector<int>& v) { std::stable_sort(v.begin(), v.end()); });
        std::printf("%-14s %12.1f %14.1f %16.1f\n", names[pattern], tb, t0, ts);
    }
    return failures == 0 ? 0 : 1;
}
