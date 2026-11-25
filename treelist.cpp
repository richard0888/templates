// TreeList — an indexed sequence (like std::vector) backed by an implicit treap.
//
//   * insert(i, x), erase(i), operator[](i)   O(log n) expected
//   * push_back / push_front / pop_back        O(log n) expected
//   * build from n values                      O(n)
//   * n operations total                       O(n log n)
//
// Nodes are keyed implicitly by their position: each node stores the size of
// its subtree, so the k-th element is found by walking down and comparing k
// with the left subtree's size. Random priorities keep the tree balanced (heap
// on priority), giving expected depth O(log n).
//
// Nodes live in a pool (std::vector) indexed by int, with a free list so erased
// slots are reused. This avoids per-node allocation and recursive destructors.
//
// Build: g++ -O2 -std=c++17 treelist.cpp -o treelist

#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

template <class T>
class TreeList {
  public:
    TreeList() = default;

    // O(n) build: Cartesian tree construction with a stack on the right spine.
    explicit TreeList(const std::vector<T>& values) {
        std::vector<int> spine;
        for (const T& v : values) {
            int n = new_node(v), last = NIL;
            while (!spine.empty() && nodes_[spine.back()].pri < nodes_[n].pri) {
                last = spine.back();
                spine.pop_back();
                pull(last);
            }
            nodes_[n].l = last;
            if (!spine.empty()) nodes_[spine.back()].r = n;
            spine.push_back(n);
        }
        root_ = spine.empty() ? NIL : spine.front();  // bottom of spine = max priority
        while (!spine.empty()) {
            pull(spine.back());
            spine.pop_back();
        }
    }

    int size() const { return sz(root_); }
    bool empty() const { return root_ == NIL; }

    // Insert x so that it ends up at index i (0 <= i <= size()).
    void insert(int i, const T& x) {
        if (i < 0 || i > size()) throw std::out_of_range("TreeList::insert");
        root_ = insert_rec(root_, i, new_node(x));
    }

    // Remove the element at index i and return it.
    T erase(int i) {
        check(i, "TreeList::erase");
        int removed = NIL;
        root_ = erase_rec(root_, i, removed);
        T out = std::move(nodes_[removed].val);
        free_.push_back(removed);
        return out;
    }

    T& operator[](int i) { return nodes_[find(i)].val; }
    const T& operator[](int i) const { return nodes_[find(i)].val; }
    T& at(int i) { check(i, "TreeList::at"); return (*this)[i]; }
    const T& at(int i) const { check(i, "TreeList::at"); return (*this)[i]; }

    void push_back(const T& x) { insert(size(), x); }
    void push_front(const T& x) { insert(0, x); }
    T pop_back() { return erase(size() - 1); }
    T pop_front() { return erase(0); }

    void clear() {
        nodes_.clear();
        free_.clear();
        root_ = NIL;
    }

    // In-order traversal, O(n), iterative.
    template <class F>
    void for_each(F&& f) const {
        std::vector<int> st;
        int cur = root_;
        while (cur != NIL || !st.empty()) {
            while (cur != NIL) { st.push_back(cur); cur = nodes_[cur].l; }
            cur = st.back();
            st.pop_back();
            f(nodes_[cur].val);
            cur = nodes_[cur].r;
        }
    }

    std::vector<T> to_vector() const {
        std::vector<T> out;
        out.reserve(size());
        for_each([&](const T& v) { out.push_back(v); });
        return out;
    }

  private:
    static constexpr int NIL = -1;

    struct Node {
        T val;
        uint32_t pri;
        int l = NIL, r = NIL, size = 1;
    };

    std::vector<Node> nodes_;
    std::vector<int> free_;
    int root_ = NIL;

    static uint32_t rand32() {
        static std::mt19937 rng(
            (uint32_t)std::chrono::steady_clock::now().time_since_epoch().count());
        return rng();
    }

    int new_node(const T& x) {
        Node n{x, rand32()};
        if (!free_.empty()) {
            int id = free_.back();
            free_.pop_back();
            nodes_[id] = std::move(n);
            return id;
        }
        nodes_.push_back(std::move(n));
        return (int)nodes_.size() - 1;
    }

    int sz(int t) const { return t == NIL ? 0 : nodes_[t].size; }
    void pull(int t) { nodes_[t].size = 1 + sz(nodes_[t].l) + sz(nodes_[t].r); }

    void check(int i, const char* who) const {
        if (i < 0 || i >= size()) throw std::out_of_range(who);
    }

    int find(int i) const {
        int t = root_;
        while (true) {
            int ls = sz(nodes_[t].l);
            if (i < ls) t = nodes_[t].l;
            else if (i == ls) return t;
            else { i -= ls + 1; t = nodes_[t].r; }
        }
    }

    // Split t into [0, k) and [k, size). Recursion depth = tree depth.
    void split(int t, int k, int& a, int& b) {
        if (t == NIL) { a = b = NIL; return; }
        int ls = sz(nodes_[t].l);
        if (k <= ls) {
            split(nodes_[t].l, k, a, nodes_[t].l);
            b = t;
        } else {
            split(nodes_[t].r, k - ls - 1, nodes_[t].r, b);
            a = t;
        }
        pull(t);
    }

    int merge(int a, int b) {
        if (a == NIL) return b;
        if (b == NIL) return a;
        if (nodes_[a].pri > nodes_[b].pri) {
            nodes_[a].r = merge(nodes_[a].r, b);
            pull(a);
            return a;
        }
        nodes_[b].l = merge(a, nodes_[b].l);
        pull(b);
        return b;
    }

    // Descend until the new node's priority wins, then split there.
    int insert_rec(int t, int i, int n) {
        if (t == NIL) return n;
        if (nodes_[n].pri > nodes_[t].pri) {
            split(t, i, nodes_[n].l, nodes_[n].r);
            pull(n);
            return n;
        }
        int ls = sz(nodes_[t].l);
        if (i <= ls) nodes_[t].l = insert_rec(nodes_[t].l, i, n);
        else nodes_[t].r = insert_rec(nodes_[t].r, i - ls - 1, n);
        pull(t);
        return t;
    }

    int erase_rec(int t, int i, int& removed) {
        int ls = sz(nodes_[t].l);
        if (i == ls) {
            removed = t;
            return merge(nodes_[t].l, nodes_[t].r);
        }
        if (i < ls) nodes_[t].l = erase_rec(nodes_[t].l, i, removed);
        else nodes_[t].r = erase_rec(nodes_[t].r, i - ls - 1, removed);
        pull(t);
        return t;
    }
};

// ---------------------------------------------------------------------------
// Tests: random ops checked against std::vector, then a timing run.

int main() {
    std::mt19937 rng(12345);

    // Correctness vs. std::vector.
    for (int round = 0; round < 50; ++round) {
        std::vector<int> init(rng() % 50);
        for (int& x : init) x = (int)(rng() % 1000);
        TreeList<int> tl(init);
        std::vector<int> ref = init;
        assert(tl.to_vector() == ref);

        for (int op = 0; op < 2000; ++op) {
            int kind = rng() % 4;
            if (kind <= 1 || ref.empty()) {
                int i = rng() % (ref.size() + 1), x = (int)(rng() % 1000);
                tl.insert(i, x);
                ref.insert(ref.begin() + i, x);
            } else if (kind == 2) {
                int i = rng() % ref.size();
                int got = tl.erase(i);
                assert(got == ref[i]);
                ref.erase(ref.begin() + i);
            } else {
                int i = rng() % ref.size(), x = (int)(rng() % 1000);
                assert(tl[i] == ref[i]);
                tl[i] = x;
                ref[i] = x;
            }
            assert(tl.size() == (int)ref.size());
        }
        assert(tl.to_vector() == ref);
    }

    // Bounds checks.
    TreeList<int> small;
    bool threw = false;
    try { small.erase(0); } catch (const std::out_of_range&) { threw = true; }
    assert(threw);

    std::puts("correctness: OK");

    // Timing: n random inserts at random positions, then n random erases.
    const int n = 1000000;
    TreeList<int> big;
    auto t0 = std::chrono::steady_clock::now();
    for (int k = 0; k < n; ++k) big.insert(rng() % (big.size() + 1), k);
    auto t1 = std::chrono::steady_clock::now();
    long long sum = 0;
    for (int k = 0; k < n; ++k) sum += big[rng() % big.size()];
    auto t2 = std::chrono::steady_clock::now();
    while (!big.empty()) big.erase(rng() % big.size());
    auto t3 = std::chrono::steady_clock::now();

    auto ms = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    std::printf("n=%d  insert %.0f ms  access %.0f ms  erase %.0f ms  (sum %lld)\n",
                n, ms(t0, t1), ms(t1, t2), ms(t2, t3), sum);
}
