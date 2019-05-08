#pragma once

#include "common.h"

constexpr size_t packed_chunk_a_offset = 3;
constexpr size_t packed_chunk_size = ch_area * 2 + packed_chunk_a_offset;

class packed_chunk
{
	std::vector<float> data;

public:
	packed_chunk() : data(packed_chunk_size, 0)
	{
	}

	float* begin()
	{
		return data.data();
	}

	float& range_begin()
	{
		return data[0];
	}

	float& range_end()
	{
		return data[1];
	}

	float& scatter_number()
	{
		return data[2];
	}

	float* a()
	{
		return &data[packed_chunk_a_offset];
	}

	float* a_line(size_t index)
	{
		return &a()[index * ch_size];
	}

	float * b()
	{
		return &data[packed_chunk_a_offset + ch_area];
	}

	float* b_line(size_t index)
	{
		return &b()[index * ch_size];
	}

};

class packed_chunk_array
{
	std::vector<float> data;

public:
	void swap(packed_chunk_array & rhs)
	{
		std::swap(data, rhs.data);
	}

	packed_chunk_array(size_t size) : data(packed_chunk_size * size, 0)
	{
	}

	void fill_0()
	{
		for (size_t i = 0; i < data.size(); ++i)
			data[i] = 0;
	}

	float* begin()
	{
		return data.data();
	}

	float* begin(size_t i)
	{
		return &data[i * packed_chunk_size];
	}

	float& range_begin(size_t i)
	{
		return data[i * packed_chunk_size];
	}

	float& range_end(size_t i)
	{
		return data[i * packed_chunk_size + 1];
	}

	float& scatter_number(size_t i)
	{
		return data[i * packed_chunk_size + 2];
	}

	float* a(size_t i)
	{
		return &data[i * packed_chunk_size + packed_chunk_a_offset];
	}

	float* a_line(size_t ch_i, size_t index)
	{
		return &a(ch_i)[index * ch_size];
	}

	float * b(size_t i)
	{
		return &data[i * packed_chunk_size + packed_chunk_a_offset + ch_area];
	}

	float* b_line(size_t ch_i, size_t index)
	{
		return &b(ch_i)[index * ch_size];
	}

};