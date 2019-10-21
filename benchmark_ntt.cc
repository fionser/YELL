#include <benchmark/benchmark.h>

#include <memory>
#include <yell/poly.hpp>
#include <nfl/poly.hpp>

class NTT : public ::benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& st) {
        p0 = std::make_shared<yell::poly<Deg>>(4, yell::uniform{});
        p1 = std::make_shared<yell::poly<Deg>>(4, yell::uniform{});
        _p0 = std::make_shared<nfl::poly<uint64_t, Deg, 4>>();
        _p1 = std::make_shared<nfl::poly<uint64_t, Deg, 4>>();
        for (int cm = 0; cm < 4; ++cm) {
            std::memcpy(_p0->begin() + Deg * cm, p0->cptr_at(cm), sizeof(uint64_t) * Deg);
            std::memcpy(_p1->begin() + Deg * cm, p1->cptr_at(cm), sizeof(uint64_t) * Deg);
        }
    }

    void TearDown(const ::benchmark::State& state) {
        if (state.thread_index == 0) {
            p0.reset();
            p1.reset();
            _p0.reset();
            _p1.reset();
        }
    }

    static constexpr size_t Deg = 8192;
    std::shared_ptr<yell::poly<Deg>> p0;
    std::shared_ptr<yell::poly<Deg>> p1;
    std::shared_ptr<nfl::poly<uint64_t, Deg, 4>> _p0;
    std::shared_ptr<nfl::poly<uint64_t, Deg, 4>> _p1;
};

BENCHMARK_F(NTT, NFLib_Forward)(benchmark::State& st) {
    for (auto _ : st)
        _p0->ntt_pow_phi();
}

BENCHMARK_F(NTT, Forward)(benchmark::State& st) {
    for (auto _ : st)
        p0->forward();
}

BENCHMARK_F(NTT, Backward)(benchmark::State& st) {
    for (auto _ : st)
        p1->forward();
}

BENCHMARK_F(NTT, NFLib_Backward)(benchmark::State& st) {
    for (auto _ : st)
        _p1->invntt_pow_invphi();
}

BENCHMARK_MAIN();
