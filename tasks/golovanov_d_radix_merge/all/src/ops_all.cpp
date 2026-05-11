#include "golovanov_d_radix_merge/all/include/ops_all.hpp"

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "golovanov_d_radix_merge/common/include/common.hpp"
#include "util/include/util.hpp"

namespace {

void MakeRadixKeys(const std::vector<double> &data, std::vector<std::uint64_t> &keys) {
  keys.resize(data.size());
  for (std::size_t i = 0; i < data.size(); ++i) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &data[i], sizeof(double));
    const std::uint64_t sign_mask = std::uint64_t{1} << 63;
    if ((bits & sign_mask) != 0) {
      bits = ~bits;
    } else {
      bits ^= sign_mask;
    }
    keys[i] = bits;
  }
}

void CountBytes(const std::vector<std::uint64_t> &keys, std::vector<std::array<std::size_t, 256>> &local_counts,
                std::size_t pass, int num_threads) {
#pragma omp parallel num_threads(num_threads)
  {
    int tid = omp_get_thread_num();
#pragma omp for
    for (std::size_t i = 0; i < keys.size(); ++i) {
      auto byte = static_cast<std::size_t>((keys[i] >> (pass * 8)) & 0xFF);
      ++local_counts[tid][byte];
    }
  }
}

void ComputeOffsets(const std::vector<std::array<std::size_t, 256>> &local_counts,
                    std::vector<std::array<std::size_t, 256>> &thread_offsets, int num_threads) {
  std::array<std::size_t, 256> total{};
  for (int t = 0; t < num_threads; ++t) {
    for (std::size_t b = 0; b < 256; ++b) {
      total[b] += local_counts[t][b];
    }
  }

  std::array<std::size_t, 256> offset{};
  std::size_t sum = 0;
  for (std::size_t b = 0; b < 256; ++b) {
    offset[b] = sum;
    sum += total[b];
  }

  for (int t = 0; t < num_threads; ++t) {
    thread_offsets[t] = offset;
    for (std::size_t b = 0; b < 256; ++b) {
      offset[b] += local_counts[t][b];
    }
  }
}

void ReorderByByte(const std::vector<double> &from, std::vector<double> &to, const std::vector<std::uint64_t> &keys,
                   const std::vector<std::array<std::size_t, 256>> &thread_offsets, int num_threads, std::size_t pass) {
#pragma omp parallel num_threads(num_threads)
  {
    int tid = omp_get_thread_num();
    auto pos = thread_offsets[tid];
#pragma omp for
    for (std::size_t i = 0; i < from.size(); ++i) {
      auto byte = static_cast<std::size_t>((keys[i] >> (pass * 8)) & 0xFF);
      to[pos[byte]++] = from[i];
    }
  }
}

void RadixSortOmp(std::vector<double> &data, int num_threads) {
  if (data.size() < 2) {
    return;
  }

  const std::size_t k_passes = sizeof(std::uint64_t);
  std::vector<double> buffer(data.size());
  std::vector<double> *from = &data;
  std::vector<double> *to = &buffer;

  std::vector<std::uint64_t> keys;

  for (std::size_t pass = 0; pass < k_passes; ++pass) {
    MakeRadixKeys(*from, keys);
    std::vector<std::array<std::size_t, 256>> local_counts(num_threads);
    CountBytes(keys, local_counts, pass, num_threads);
    std::vector<std::array<std::size_t, 256>> thread_offsets(num_threads);
    ComputeOffsets(local_counts, thread_offsets, num_threads);
    ReorderByByte(*from, *to, keys, thread_offsets, num_threads, pass);
    std::swap(from, to);
  }

  if (from != &data) {
    data = *from;
  }
}

void MergeSortedVectors(std::vector<double> &left, const std::vector<double> &right) {
  if (right.empty()) {
    return;
  }
  if (left.empty()) {
    left = right;
    return;
  }
  std::vector<double> merged(left.size() + right.size());
  std::merge(left.begin(), left.end(), right.begin(), right.end(), merged.begin(), [](double a, double b) {
    std::uint64_t ka = 0, kb = 0;
    std::memcpy(&ka, &a, sizeof(double));
    std::memcpy(&kb, &b, sizeof(double));
    const std::uint64_t mask = std::uint64_t{1} << 63;
    if ((ka & mask) != 0) {
      ka = ~ka;
    } else {
      ka ^= mask;
    }
    if ((kb & mask) != 0) {
      kb = ~kb;
    } else {
      kb ^= mask;
    }
    return ka < kb;
  });
  left.swap(merged);
}

}  // namespace

namespace golovanov_d_radix_merge {

GolovanovDRadixMergeALL::GolovanovDRadixMergeALL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool GolovanovDRadixMergeALL::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int is_valid = 1;
  if (rank == 0) {
    is_valid = !GetInput().empty() ? 1 : 0;
  }
  MPI_Bcast(&is_valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return is_valid != 0;
}

bool GolovanovDRadixMergeALL::PreProcessingImpl() {
  return true;
}

bool GolovanovDRadixMergeALL::RunImpl() {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const int num_threads = ppc::util::GetNumThreads();
  uint64_t global_size = 0;
  if (rank == 0) {
    global_size = GetInput().size();
  }
  MPI_Bcast(&global_size, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
  const int global_size_int = static_cast<int>(global_size);
  std::vector<int> send_counts(size, 0);
  std::vector<int> displs(size, 0);
  const int base_count = global_size_int / size;
  const int remainder = global_size_int % size;
  int offset = 0;
  for (int proc = 0; proc < size; ++proc) {
    send_counts[proc] = base_count + (proc < remainder ? 1 : 0);
    displs[proc] = offset;
    offset += send_counts[proc];
  }

  std::vector<double> local_data(send_counts[rank]);
  double *send_buffer = nullptr;
  if (rank == 0) {
    send_buffer = GetInput().data();
  }
  MPI_Scatterv(send_buffer, send_counts.data(), displs.data(), MPI_DOUBLE,
               local_data.empty() ? nullptr : local_data.data(), send_counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

  RadixSortOmp(local_data, num_threads);

  for (int step = 1; step < size; step <<= 1) {
    const int group_size = step << 1;
    if ((rank % group_size) == 0) {
      const int src = rank + step;
      if (src < size) {
        int recv_count = 0;
        MPI_Recv(&recv_count, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (recv_count > 0) {
          std::vector<double> received_data(recv_count);
          MPI_Recv(received_data.data(), recv_count, MPI_DOUBLE, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          MergeSortedVectors(local_data, received_data);
        }
      }
    } else {
      const int dst = rank - step;
      const int send_count = static_cast<int>(local_data.size());
      if (send_count > 0) {
        MPI_Send(&send_count, 1, MPI_INT, dst, 0, MPI_COMM_WORLD);
        MPI_Send(local_data.data(), send_count, MPI_DOUBLE, dst, 1, MPI_COMM_WORLD);
      }
      local_data.clear();
      break;
    }
  }

  std::vector<double> result(global_size_int);
  if (rank == 0) {
    result.swap(local_data);
  }
  MPI_Bcast(result.data(), global_size_int, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  GetOutput() = std::move(result);
  return true;
}

bool GolovanovDRadixMergeALL::PostProcessingImpl() {
  return true;
}

}  // namespace golovanov_d_radix_merge
