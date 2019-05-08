#pragma once

#include <vector>

constexpr size_t ch_size = 1047;
constexpr size_t ch_area = ch_size * ch_size;



class packed_result
{
	std::vector<float> data_;
public:
	packed_result()
	{
		data_.resize(ch_area,0);
	}

	void fill_0()
	{
		for (size_t i = 0; i < data_.size(); ++i)
			data_[i] = 0;
	}

	float* data()
	{
		return data_.data();
	}
};

using res_t = packed_result;