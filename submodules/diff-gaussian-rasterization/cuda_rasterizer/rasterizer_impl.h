/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#pragma once

#include <iostream>
#include <vector>
#include "rasterizer.h"
#include <cuda_runtime_api.h>

namespace CudaRasterizer
{
	/**
	 * Create and allocate T-type pointers from a block of bare byte pointers
	 * The use of alignment = 2^n, the role of the chunk plus alignment-1 after the low n position zero,
	 * that is, the alignment of the unit unconditional into one, thus align memory allocation,
	 * so that the ptr and chunk to alignment byte unit chunks
	 */
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T*>(offset); // Output result: allocate memory start location to member pointer
		chunk = reinterpret_cast<char*>(ptr + count); // First ptr+count counts the memory location where the next member should start, then converts it to a pointer of the generic transition type
	}

	/**
	 * state of 3D point cloud
	 */
	struct GeometryState
	{
		size_t scan_size;
		float* depths;
		char* scanning_space;
		bool* clamped;
		int* internal_radii;
		float2* means2D;
		float* cov3D;
		float4* conic_opacity;
		float* rgb;
		uint32_t* point_offsets;
		uint32_t* tiles_touched;

		static GeometryState fromChunk(char*& chunk, size_t P);
	};

	/**
	 * state of image pixels
	 */
	struct ImageState
	{
		uint2* ranges; // the ordinal number of the first Gaus instance corresponding to a tile (stored in .x) and the ordinal number of the first Gaus instance of the next tile (stored in .y) (after sorting); so many pixels are reserved, but only so many tiles are actually needed?
		uint32_t* n_contrib;
		float* accum_alpha; // rendering of a tile is complete when all pixels in the tile have a cumulative alpha value of 1.

		static ImageState fromChunk(char*& chunk, size_t N);
	};

	/**
	 * 
	 */
	struct BinningState
	{
		size_t sorting_size;
		uint64_t* point_list_keys_unsorted;
		uint64_t* point_list_keys;
		uint32_t* point_list_unsorted;
		uint32_t* point_list;
		char* list_sorting_space;

		static BinningState fromChunk(char*& chunk, size_t P);
	};

	/**
	 * Get the size of the storage space required when T (three states) has P elements
	 * size = nullptr starts from zero, so the result is a direct cumulative quantity (plus alignment)
	 */
	template<typename T> 
	size_t required(size_t P)
	{
		char* size = nullptr;
		T::fromChunk(size, P);
		return ((size_t)size) + 128;
	}
};