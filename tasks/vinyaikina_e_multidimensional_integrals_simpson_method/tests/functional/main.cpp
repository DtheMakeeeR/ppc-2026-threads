#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stack>
#include <vector>
#include <numbers>

#include "vinyaikina_e_multidimensional_integrals_simpson_method/common/include/common.hpp"
#include "vinyaikina_e_multidimensional_integrals_simpson_method/omp/include/ops_omp.hpp"
#include "vinyaikina_e_multidimensional_integrals_simpson_method/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace vinyaikina_e_multidimensional_integrals_simpson_method {
  namespace {

    class VinyaikinaESimpsonFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
    public:
      static std::string PrintTestParam(const TestType& test_param) {
        return std::get<0>(test_param);
      }

    protected:
      void SetUp() override {
        TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
        input_ = std::get<1>(params);
        etalon_ = std::get<2>(params);

      }

      bool CheckTestOutputData(OutType& output_data) final {
        const double eps = 1e-3;
        return std::fabs(output_data - etalon_) <= eps;
      }

      InType GetTestInputData() final {
        return input_;
      }

    private:
      InType input_;
      OutType etalon_ = 0.0;
    };

    TEST_P(VinyaikinaESimpsonFuncTests, Run) {
      ExecuteTest(GetParam());
    }

    double count_n_dim_area(const std::vector<std::pair<double, double>>& borders) {
      double area = 1.0;
      for (size_t i = 0; i < borders.size(); i++) {
        area *= (borders[i].second - borders[i].first);
      }
      return area;
    }

    double int_linear_1d(double a, double b) {
      return (b * b - a * a) / 2.0;
    }

    double int_sin_1d(double a, double b) {
      return std::cos(a) - std::cos(b);
    }

    double int_x2_1d(double a, double b) {
      return (b * b * b - a * a * a) / 3.0;
    }

    double int_x3_1d(double a, double b) {
      return (b * b * b * b - a * a * a * a) / 4.0;
    }

    double int_x4_1d(double a, double b) {
      return (b * b * b * b * b - a * a * a * a * a) / 5.0;
    }

    double int_exp_1d(double a, double b) {
      return std::exp(b) - std::exp(a);
    }

    double int_cos_1d(double a, double b) {
      return std::sin(b) - std::sin(a);
    }

    double int_xy_2d(double a1, double b1, double a2, double b2) {
      return ((b1 * b1 - a1 * a1) * (b2 * b2 - a2 * a2)) / 4.0;
    }

    double int_x_plus_y_2d(double a1, double b1, double a2, double b2) {
      return (b2 - a2) * (b1 * b1 - a1 * a1) / 2.0 +
        (b1 - a1) * (b2 * b2 - a2 * a2) / 2.0;
    }

    double int_x2_y_2d(double a1, double b1, double a2, double b2) {
      return int_x2_1d(a1, b1) * int_linear_1d(a2, b2);
    }

    double int_xyz_3d(double a1, double b1, double a2, double b2, double a3, double b3) {
      return int_linear_1d(a1, b1) * int_linear_1d(a2, b2) * int_linear_1d(a3, b3);
    }

    double int_x2_y2_z2_3d(double a1, double b1, double a2, double b2, double a3, double b3) {
      return int_x2_1d(a1, b1) * int_x2_1d(a2, b2) * int_x2_1d(a3, b3);
    }

    double int_exp_sum_3d(double a1, double b1, double a2, double b2, double a3, double b3) {
      return int_exp_1d(a1, b1) * int_exp_1d(a2, b2) * int_exp_1d(a3, b3);
    }

    auto one = [](const std::vector<double>&) { return 1.0; };
    auto linear_1d = [](const std::vector<double>& x) { return x[0]; };
    auto sin_1d = [](const std::vector<double>& x) { return std::sin(x[0]); };
    auto x2_1d = [](const std::vector<double>& x) { return x[0] * x[0]; };
    auto x3_1d = [](const std::vector<double>& x) { return x[0] * x[0] * x[0]; };
    auto x4_1d = [](const std::vector<double>& x) { return std::pow(x[0], 4); };
    auto exp_1d = [](const std::vector<double>& x) { return std::exp(x[0]); };
    auto cos_1d = [](const std::vector<double>& x) { return std::cos(x[0]); };

    auto xy_2d = [](const std::vector<double>& x) { return x[0] * x[1]; };
    auto x_plus_y_2d = [](const std::vector<double>& x) { return x[0] + x[1]; };
    auto x2_y_2d = [](const std::vector<double>& x) { return x[0] * x[0] * x[1]; };

    auto xyz_3d = [](const std::vector<double>& x) { return x[0] * x[1] * x[2]; };
    auto x2_y2_z2_3d = [](const std::vector<double>& x) {
      return x[0] * x[0] * x[1] * x[1] * x[2] * x[2];
      };
    auto exp_sum_3d = [](const std::vector<double>& x) {
      return std::exp(x[0] + x[1] + x[2]);
      };

    const std::array<TestType, 16> kTests = { {
        TestType{"area_1d_0_1",
                 InType{0.005, {{0.0, 1.0}}, one},
                 count_n_dim_area({{0.0, 1.0}})},

        TestType{"area_2d_0_1_x_0_1",
                 InType{0.005, {{0.0, 1.0}, {0.0, 1.0}}, one},
                 count_n_dim_area({{0.0, 1.0}, {0.0, 1.0}})},

        TestType{"volume_3d_0_05_x3",
                 InType{0.005, {{0.0, 0.5}, {0.0, 0.5}, {0.0, 0.5}}, one},
                 count_n_dim_area({{0.0, 0.5}, {0.0, 0.5}, {0.0, 0.5}})},

        TestType{"linear_1d_0_2",
                 InType{0.01, {{0.0, 2.0}}, linear_1d},
                 int_linear_1d(0.0, 2.0)},

        TestType{"x2_1d_0_1",
                 InType{0.005, {{0.0, 1.0}}, x2_1d},
                 int_x2_1d(0.0, 1.0)},

        TestType{"x3_1d_0_1",
                 InType{0.005, {{0.0, 1.0}}, x3_1d},
                 int_x3_1d(0.0, 1.0)},

        TestType{"x4_1d_0_1",
                 InType{0.005, {{0.0, 1.0}}, x4_1d},
                 int_x4_1d(0.0, 1.0)},

        TestType{"exp_1d_0_1",
                 InType{0.005, {{0.0, 1.0}}, exp_1d},
                 int_exp_1d(0.0, 1.0)},

        TestType{"cos_1d_0_pi2",
                 InType{0.001, {{0.0, std::numbers::pi / 2.0}}, cos_1d},
                 int_cos_1d(0.0, std::numbers::pi / 2.0)},

        TestType{"sin_1d_0_pi",
                 InType{0.001, {{0.0, std::numbers::pi}}, sin_1d},
                 int_sin_1d(0.0, std::numbers::pi)},

        TestType{"xy_2d_0_1_x_0_1",
                 InType{0.005, {{0.0, 1.0}, {0.0, 1.0}}, xy_2d},
                 int_xy_2d(0.0, 1.0, 0.0, 1.0)},

        TestType{"x_plus_y_2d_0_1_x_0_1",
                 InType{0.005, {{0.0, 1.0}, {0.0, 1.0}}, x_plus_y_2d},
                 int_x_plus_y_2d(0.0, 1.0, 0.0, 1.0)},

        TestType{"x2_y_2d_0_1_x_0_1",
                 InType{0.005, {{0.0, 1.0}, {0.0, 1.0}}, x2_y_2d},
                 int_x2_y_2d(0.0, 1.0, 0.0, 1.0)},

        TestType{"xyz_3d_0_1_x3",
                 InType{0.005, {{0.0, 0.75}, {0.0, 0.75}, {0.0, 0.75}}, xyz_3d},
                 int_xyz_3d(0.0, 0.75, 0.0, 0.75, 0.0, 0.75)},

        TestType{"x2_y2_z2_3d_0_1_x3",
                 InType{0.01, {{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}}, x2_y2_z2_3d},
                 int_x2_y2_z2_3d(0.0, 1.0, 0.0, 1.0, 0.0, 1.0)},

        TestType{"exp_sum_3d_0_05_x3",
                 InType{0.005, {{0.0, 0.5}, {0.0, 0.5}, {0.0, 0.5}}, exp_sum_3d},
                 int_exp_sum_3d(0.0, 0.5, 0.0, 0.5, 0.0, 0.5)},
    } };

    const auto kTaskName = PPC_SETTINGS_vinyaikina_e_multidimensional_integrals_simpson_method;

    const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<VinyaikinaEMultidimIntegrSimpsonSEQ, InType>(
      kTests, kTaskName), ppc::util::AddFuncTask<VinyaikinaEMultidimIntegrSimpsonOMP, InType>(
        kTests, kTaskName));

    const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

    const auto kFuncTestName = VinyaikinaESimpsonFuncTests::PrintFuncTestName<VinyaikinaESimpsonFuncTests>;

    INSTANTIATE_TEST_SUITE_P(MultidinIntegralsSimpsonTests, VinyaikinaESimpsonFuncTests, kGtestValues, kFuncTestName);

  } // namespace 
} // namespace vinyaikina_e_multidimensional_integrals_simpson_method
