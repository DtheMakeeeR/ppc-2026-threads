#include "nikolaev_d_block_linear_image_filtering_seq/seq/include/ops_seq.hpp"

#include <algorithm>
#include <vector>
#include <array>
#include <cstdint>

#include "nikolaev_d_block_linear_image_filtering_seq/common/include/common.hpp"
#include "util/include/util.hpp"

namespace nikolaev_d_block_linear_image_filtering {

NikolaevDBlockLinearImageFilteringSEQ::NikolaevDBlockLinearImageFilteringSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<uint8_t>();
}

bool NikolaevDBlockLinearImageFilteringSEQ::ValidationImpl() {
  const auto img_width = get<0>(GetInput());
  const auto img_height = get<1>(GetInput());
  const auto &pixel_data = get<2>(GetInput());

  return static_cast<std::size_t>(img_width) * static_cast<std::size_t>(img_height) * 3 == pixel_data.size();
}

bool NikolaevDBlockLinearImageFilteringSEQ::PreProcessingImpl() {
  return true;
}

std::uint8_t NikolaevDBlockLinearImageFilteringSEQ::GetPixel(const std::vector<uint8_t>& data, int w, int h, int x, int y, int ch) {
    int ix = std::clamp(x, 0, w - 1);
    int iy = std::clamp(y, 0, h - 1);
    return data[(iy * w + ix) * 3 + ch];
}

bool NikolaevDBlockLinearImageFilteringSEQ::RunImpl() {
  const int width = std::get<0>(GetInput());
  const int height = std::get<1>(GetInput());
  const auto& src = std::get<2>(GetInput());
  
  auto& dst = GetOutput();
  dst.assign(src.size(), 0);

  const int kernel[3][3] = {
      {1, 2, 1},
      {2, 4, 2},
      {1, 2, 1}
  };
  const int kSum = 16;

  for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
          for (int ch = 0; ch < 3; ++ch) {
              
              int acc = 0;
              for (int ky = -1; ky <= 1; ++ky) {
                  for (int kx = -1; kx <= 1; ++kx) {
                      acc += GetPixel(src, width, height, x + kx, y + ky, ch) * kernel[ky + 1][kx + 1];
                  }
              }

              int res = (acc + 8) / kSum;
              dst[(y * width + x) * 3 + ch] = static_cast<uint8_t>(std::clamp(res, 0, 255));
          }
      }
  }
  return true;
}

bool NikolaevDBlockLinearImageFilteringSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace nikolaev_d_block_linear_image_filtering
