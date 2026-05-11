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

void RadixSortOmp(std::vector<double> &data, int num_threads) {
  if (data.size() < 2) {
    return;
  }

  const std::size_t k_base = 256;
  const std::size_t k_passes = sizeof(std::uint64_t);

  std::vector<double> buffer(data.size());
  std::vector<double> *from = &data;
  std::vector<double> *to = &buffer;

  const std::size_t n = data.size();

  std::vector<std::array<std::size_t, k_base>> local_counts(num_threads);
  std::vector<std::array<std::size_t, k_base>> thread_offsets(num_threads);

  for (std::size_t pass = 0; pass < k_passes; ++pass) {
    for (auto &cnt : local_counts) {
      cnt.fill(0);
    }

#pragma omp parallel num_threads(num_threads) default(none) \
    shared(from, to, local_counts, thread_offsets, n, pass, num_threads, k_base) private(tid)
    {
      int tid = omp_get_thread_num();

#pragma omp for schedule(static)
      for (std::size_t i = 0; i < n; ++i) {
        std::uint64_t bits = 0;
        std::memcpy(&bits, &(*from)[i], sizeof(double));
        const std::uint64_t sign_mask = std::uint64_t{1} << 63;
        if (bits & sign_mask) {
          bits = ~bits;
        } else {
          bits ^= sign_mask;
        }
        const std::size_t byte = static_cast<std::size_t>((bits >> (pass * 8)) & 0xFF);
        ++local_counts[tid][byte];
      }

#pragma omp single
      {
        std::array<std::size_t, k_base> total{};
        total.fill(0);

        for (int thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
          for (std::size_t b_idx = 0; b_idx < k_base; ++b_idx) {
            total[b_idx] += local_counts[thread_idx][b_idx];
          }
        }

        std::array<std::size_t, k_base> offset{};
        std::size_t sum = 0;
        for (std::size_t b_idx = 0; b_idx < k_base; ++b_idx) {
          offset[b_idx] = sum;
          sum += total[b_idx];
        }

        for (int thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
          thread_offsets[thread_idx] = offset;
          for (std::size_t b_idx = 0; b_idx < k_base; ++b_idx) {
            offset[b_idx] += local_counts[thread_idx][b_idx];
          }
        }
      }

      auto pos = thread_offsets[tid];

#pragma omp for schedule(static)
      for (std::size_t i = 0; i < n; ++i) {
        std::uint64_t bits = 0;
        std::memcpy(&bits, &(*from)[i], sizeof(double));
        const std::uint64_t sign_mask = std::uint64_t{1} << 63;
        if (bits & sign_mask) {
          bits = ~bits;
        } else {
          bits ^= sign_mask;
        }
        const std::size_t byte = static_cast<std::size_t>((bits >> (pass * 8)) & 0xFF);
        (*to)[pos[byte]++] = (*from)[i];
      }
    }

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
    std::uint64_t ka = 0;
    std::uint64_t kb = 0;
    std::memcpy(&ka, &a, sizeof(double));
    std::memcpy(&kb, &b, sizeof(double));
    const std::uint64_t mask = std::uint64_t{1} << 63;
    if (ka & mask) {
      ka = ~ka;
    } else {
      ka ^= mask;
    }
    if (kb & mask) {
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
  int rank = 0;
  int size = 1;
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
        std::vector<double> received_data(recv_count);
        MPI_Recv(received_data.data(), recv_count, MPI_DOUBLE, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MergeSortedVectors(local_data, received_data);
      }
    } else {
      const int dst = rank - step;
      const int send_count = static_cast<int>(local_data.size());
      MPI_Send(&send_count, 1, MPI_INT, dst, 0, MPI_COMM_WORLD);
      MPI_Send(local_data.data(), send_count, MPI_DOUBLE, dst, 1, MPI_COMM_WORLD);
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
