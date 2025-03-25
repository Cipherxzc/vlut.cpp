#include <bits/stdc++.h>
using namespace std;
using LL = long long;

class BitNetNum {
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

    pair<int, int> upbit() const {
        for (int i = bits_.size() - 1; i >= 0; i--) {
            if (bits_[i] == 1) {
                return {i, 1};
            } else if (bits_[i] == -1) {
                return {i, -1};
            }
        }
        return {-1, 0};
    }

    int toI2() const {
        int res = 0;
        for (int i = bits_.size() - 1; i >= 0; i--) {
            int tmp = bits_[i];
            if (tmp == -1) {
                tmp = 3;
            }
            res = res * 4 + tmp;
        }
        return res;
    }

    int toI1_58() const {
        int res = 0;
        for (int i = bits_.size() - 1; i >= 0; i--) {
            res = res * 3 + bits_[i] + 1;
        }
        return res;
    }

    int toI2S() const { return toI1_58(); }

    static BitNetNum fromI2(int x) {
        BitNetNum res;
        for (int i = 0; i < 4; i++) {
            int gg = x % 4;
            x /= 4;
            if (gg == 3) {
                gg = -1;
            }
            res.bits_.push_back(gg);
        }
        return res;
    }

    static BitNetNum fromI1_58(int x) {
        BitNetNum res;
        for (int i = 0; i < 5; i++) {
            int gg = x % 3 - 1;
            x /= 3;
            res.bits_.push_back(gg);
        }
        return res;
    }

    static BitNetNum fromI2S(int x) {
        BitNetNum res;
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

inline void rev(int x, int y) { cout << "    rev(table + " << x << " * nr, table + " << y << " * nr, nr);\n"; }
inline void add(int x, int y, int pos) {
    cout << "    add(table + " << x << " * nr, table + " << y << " * nr, y" << pos << ", nr);\n";
}
inline void sub(int x, int y, int pos) {
    cout << "    sub(table + " << x << " * nr, table + " << y << " * nr, y" << pos << ", nr);\n";
}

inline void rev_tile(int x, int y) {
    cout << "    rev_tile(table + " << x << " * TABLE_ENTRY_SIZE, table + " << y << " * TABLE_ENTRY_SIZE);\n";
}
inline void add_tile(int x, int y, int pos) {
    cout << "    add_tile(table + " << x << " * TABLE_ENTRY_SIZE, table + " << y << " * TABLE_ENTRY_SIZE, y" << pos
         << ");\n";
}
inline void sub_tile(int x, int y, int pos) {
    cout << "    sub_tile(table + " << x << " * TABLE_ENTRY_SIZE, table + " << y << " * TABLE_ENTRY_SIZE, y" << pos
         << ");\n";
}

void generate_function(int s, string name) {
    cout << "void " << name << "(int16_t *restrict table, const int8_t *restrict y, int nr) {\n";
    cout << "    const int8_t *restrict y0 = y;\n";
    cout << "    const int8_t *restrict y1 = y0 + nr;\n";
    cout << "    const int8_t *restrict y2 = y1 + nr;\n";
    cout << "    const int8_t *restrict y3 = y2 + nr;\n\n";

    queue<int> que;
    que.push(s);
    while (!que.empty()) {
        int p = que.front();
        que.pop();
        for (auto [q, type] : e[p]) {
            if (type == 0) {
                rev(q, p);
            } else if (type > 0) {
                add(q, p, type - 1);
            } else if (type < 0) {
                sub(q, p, -type - 1);
            }
            que.push(q);
        }
    }

    cout << "}\n";
}

void generate_function_tile(int s, string name) {
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
            if (type == 0) {
                rev_tile(q, p);
            } else if (type > 0) {
                add_tile(q, p, type - 1);
            } else if (type < 0) {
                sub_tile(q, p, -type - 1);
            }
            que.push(q);
        }
    }

    cout << "}\n";
}

void gemm_make_table_I2() {
    for (int i = 1; i < 256; i++) {
        BitNetNum x = BitNetNum::fromI2(i), y = x;
        if (!x.validate()) {
            continue;
        }
        if (x.upbit().second == -1 && false) {
            y.reverse();
            add_edge(x.toI2(), y.toI2(), 0);
        } else {
            auto [pos, val] = x.lowbit();
            if (val == 1) {
                y.bits_[pos] = 0;
                add_edge(x.toI2(), y.toI2(), pos + 1);
            } else if (val == -1) {
                y.bits_[pos] = 0;
                add_edge(x.toI2(), y.toI2(), -pos - 1);
            } else {
                assert(0);
            }
        }
    }

    // generate_function(0, "gemm_make_table_I2");
    generate_function_tile(0, "gemm_make_table_I2_tile");
}

void gemm_make_table_I2S() {
    for (int i = 0; i < 81; i++) {
        BitNetNum x = BitNetNum::fromI2S(i), y = x;
        if (!x.validate() || i == 40) {
            continue;
        }
        if (x.upbit().second == -1) {
            y.reverse();
            add_edge(x.toI2S(), y.toI2S(), 0);
        } else {
            auto [pos, val] = x.lowbit();
            if (val == 1) {
                y.bits_[pos] = 0;
                add_edge(x.toI2S(), y.toI2S(), pos + 1);
            } else if (val == -1) {
                y.bits_[pos] = 0;
                add_edge(x.toI2S(), y.toI2S(), -pos - 1);
            } else {
                assert(0);
            }
        }
    }

    // generate_function(40, "gemm_make_table_I2S");
    generate_function_tile(40, "gemm_make_table_I2S_tile");
}

int main() {
    gemm_make_table_I2();
    // gemm_make_table_I2S();

    return 0;
}