#include "golovanov_d_radix_merge/all/include/ops_all.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <utility>
#include <vector>

#include "../include/ops_all.hpp"
#include "../include/radix_sort_all.hpp"
#include "golovanov_d_radix_merge/common/include/common.hpp"

namespace golovanov_d_radix_merge {

GolovanovDRadixMergeALL::GolovanovDRadixMergeALL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool GolovanovDRadixMergeALL::ValidationImpl() {
  return !GetInput().empty();
}

bool GolovanovDRadixMergeALL::PreProcessingImpl() {
  return true;
}

bool GolovanovDRadixMergeALL::RunImpl() {
  // ПОДГОТОВКА ПРИКОЛОВ
  const std::size_t n = GetInput().size();

  int mpi_rank = 0;
  int mpi_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  std::size_t local_size = n / static_cast<std::size_t>(mpi_size);
  std::size_t remainder = n % static_cast<std::size_t>(mpi_size);

  std::size_t start = mpi_rank * local_size + std::min<std::size_t>(static_cast<std::size_t>(mpi_rank), remainder);
  std::size_t end = start + local_size + (static_cast<std::size_t>(mpi_rank) < remainder ? 1U : 0U);

  std::vector<double> local_data(GetInput().begin() + static_cast<std::ptrdiff_t>(start),
                                 GetInput().begin() + static_cast<std::ptrdiff_t>(end));

  // ВЫЗОВ ФУНКЦИИ
  RadixSortALL::Sort(local_data);

  // СБРОС НА РУТ
  std::vector<int> recv_counts(mpi_size);
  std::vector<int> displs(mpi_size);
  int local_count = static_cast<int>(local_data.size());

  MPI_Gather(&local_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (mpi_rank == 0) {
    displs[0] = 0;
    for (int i = 1; i < mpi_size; ++i) {
      displs[i] = displs[i - 1] + recv_counts[i - 1];
    }
    GetOutput().resize(n);
  }

  MPI_Gatherv(local_data.data(), local_count, MPI_DOUBLE, GetOutput().data(), recv_counts.data(), displs.data(),
              MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // МЁЁРЖ
  if (mpi_rank == 0) {
    std::vector<std::vector<double>> parts(mpi_size);
    for (int i = 0; i < mpi_size; ++i) {
      parts[i] = std::vector<double>(GetOutput().begin() + displs[i], GetOutput().begin() + displs[i] + recv_counts[i]);
    }
    while (parts.size() > 1) {
      std::vector<std::vector<double>> next((parts.size() + 1) / 2);
      for (std::size_t i = 0; i < parts.size() / 2; ++i) {
        next[i] = RadixSortALL::Merge(parts[2 * i], parts[2 * i + 1]);
      }
      if (parts.size() % 2 != 0) {
        next.back() = parts.back();
      }
      parts = std::move(next);
    }
    if (!parts.empty()) {
      GetOutput() = std::move(parts[0]);
    }
  }

  return true;
}

bool GolovanovDRadixMergeALL::PostProcessingImpl() {
  return true;
}

}  // namespace golovanov_d_radix_merge
