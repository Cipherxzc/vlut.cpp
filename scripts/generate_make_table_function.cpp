#include <bits/stdc++.h>
using namespace std;
using LL = long long;

class VLutNum {
   public:
    vector<int> bits_;

    void reverse() {
        for (int &x : bits_) {
            x = -x;
        }
    }

    bool validate() const {
        for (int x : bits_) {
            if (x != 0 && x != 1 && x != -1) {
                return false;
            }
        }
        return true;
    }

    pair<int, int> lowbit() const {
        for (int i = 0; i < bits_.size(); i++) {
            if (bits_[i] == 1) {
                return {i, 1};
            } else if (bits_[i] == -1) {
                return {i, -1};
            }
        }
        return {-1, 0};
    }

    int toI1_58() const {
        int res = 0;
        for (int i = bits_.size() - 1; i >= 0; i--) {
            res = res * 3 + bits_[i] + 1;
        }
        return res;
    }

    int toI2S() const { return toI1_58(); }

    static VLutNum fromI1_58(int x) {
        VLutNum res;
        for (int i = 0; i < 5; i++) {
            int gg = x % 3 - 1;
            x /= 3;
            res.bits_.push_back(gg);
        }
        return res;
    }

    static VLutNum fromI2S(int x) {
        VLutNum res;
        for (int i = 0; i < 4; i++) {
            int gg = x % 3 - 1;
            x /= 3;
            res.bits_.push_back(gg);
        }
        return res;
    }
};

const int N = 1005;
vector<pair<int, int>> e[N];

inline void add_edge(int u, int v, int type) { e[v].push_back({u, type}); }

inline void rev(int x, int y) {
    cout << "    rev(table + " << x << " * TABLE_ENTRY_SIZE, table + " << y << " * TABLE_ENTRY_SIZE);\n";
}
inline void add(int x, int y, int pos) {
    cout << "    add(table + " << x << " * TABLE_ENTRY_SIZE, table + " << y << " * TABLE_ENTRY_SIZE, y" << pos
         << ");\n";
}

void generate_function(int s, string name) {
    cout << "void " << name << "(int16_t *restrict table, const int8_t *restrict y) {\n";
    cout << "    const int8_t *restrict y0 = y;\n";
    cout << "    const int8_t *restrict y1 = y0 + TABLE_ENTRY_SIZE;\n";
    cout << "    const int8_t *restrict y2 = y1 + TABLE_ENTRY_SIZE;\n";
    cout << "    const int8_t *restrict y3 = y2 + TABLE_ENTRY_SIZE;\n\n";

    queue<int> que;
    que.push(s);
    while (!que.empty()) {
        int p = que.front();
        que.pop();
        for (auto [q, type] : e[p]) {
            if (type == -1) {
                rev(q, p);
            } else {
                assert(type >= 0);
                add(q, p, type);
            }
            que.push(q);
        }
    }

    cout << "}\n";
}

void gemm_make_table_I2S() {
    for (int i = 0; i < 81; i++) {
        VLutNum x = VLutNum::fromI2S(i), y = x;
        if (!x.validate() || i == 40) {
            continue;
        }
        auto [pos, val] = x.lowbit();
        if (val == -1) {
            y.reverse();
            add_edge(x.toI2S(), y.toI2S(), -1);
        } else if (val == 1) {
            y.bits_[pos] = 0;
            add_edge(x.toI2S(), y.toI2S(), pos);
        } else {
            assert(0);
        }
    }

    generate_function(40, "gemm_make_table_i2s");
}

void gemm_make_table_I1_58S() {
    for (int i = 0; i < 243; i++) {
        VLutNum x = VLutNum::fromI1_58(i), y = x;
        if (!x.validate() || i == 121) {
            continue;
        }
        auto [pos, val] = x.lowbit();
        if (val == -1) {
            y.reverse();
            add_edge(x.toI1_58(), y.toI1_58(), -1);
        } else if (val == -1) {
            y.bits_[pos] = 0;
            add_edge(x.toI1_58(), y.toI1_58(), pos);
        } else {
            assert(0);
        }
    }

    generate_function(121, "gemm_make_table_i1_58s");
}

int main() {
    gemm_make_table_I2S();
    // gemm_make_table_I1_58S();

    return 0;
}