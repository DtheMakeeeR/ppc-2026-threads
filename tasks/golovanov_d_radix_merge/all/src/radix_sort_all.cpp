#include "../include/radix_sort_all.hpp"

#include <omp.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace golovanov_d_radix_merge {

namespace {

constexpr int kBytes = 8;
constexpr std::size_t kRadix = 256;
constexpr std::uint64_t kSignMask = 1ULL << 63;
constexpr std::uint64_t kByteMask = 0xFFULL;

std::uint64_t ToSortable(std::uint64_t bits) {
  return ((bits & kSignMask) != 0U) ? ~bits : (bits ^ kSignMask);
}

std::uint64_t FromSortable(std::uint64_t bits) {
  return ((bits & kSignMask) != 0U) ? (bits ^ kSignMask) : ~bits;
}

}  // namespace

void RadixSortALL::SortRange(std::vector<double> &arr, std::size_t left, std::size_t right) {
  if (right <= left) {
    return;
  }

  std::size_t n = right - left;
  std::vector<std::uint64_t> data(n);

  for (int byte = 0; byte < kBytes; ++byte) {
    std::array<std::size_t, kRadix> count{};

// Преобразование в sortable
#pragma omp parallel for default(none) shared(data, n, arr, count) firstprivate(byte, left)
    for (std::size_t i = 0; i < n; ++i) {
      std::uint64_t bits = 0;
      std::memcpy(&bits, &arr[left + i], sizeof(double));
      data[i] = ToSortable(bits);
    }

    // Счётчик по байтам
    for (std::size_t i = 0; i < n; ++i) {
      auto b = static_cast<std::size_t>((data[i] >> (byte * 8)) & kByteMask);
      ++count.at(b);
    }

    std::size_t sum = 0;
    for (std::size_t i = 0; i < kRadix; ++i) {
      auto tmp = count.at(i);
      count.at(i) = sum;
      sum += tmp;
    }

    std::vector<std::uint64_t> buffer(n);
    for (std::size_t i = 0; i < n; ++i) {
      auto b = static_cast<std::size_t>((data[i] >> (byte * 8)) & kByteMask);
      auto pos = count.at(b);
      buffer[pos] = data[i];
      ++count.at(b);
    }

    data.swap(buffer);
  }

// Преобразование обратно в double
#pragma omp parallel for default(none) shared(data, arr) firstprivate(left, n)
  for (std::size_t i = 0; i < n; ++i) {
    std::uint64_t bits = FromSortable(data[i]);
    std::memcpy(&arr[left + i], &bits, sizeof(double));
  }
}

std::vector<double> RadixSortALL::Merge(const std::vector<double> &a, const std::vector<double> &b) {
  std::vector<double> result;
  result.reserve(a.size() + b.size());
  std::size_t i = 0;
  std::size_t j = 0;

  while (i < a.size() && j < b.size()) {
    if (a[i] <= b[j]) {
      result.push_back(a[i++]);
    } else {
      result.push_back(b[j++]);
    }
  }
  while (i < a.size()) {
    result.push_back(a[i++]);
  }
  while (j < b.size()) {
    result.push_back(b[j++]);
  }

  return result;
}

void RadixSortALL::Sort(std::vector<double> &arr) {
  SortRange(arr, 0, arr.size());
}

}  // namespace golovanov_d_radix_merge
