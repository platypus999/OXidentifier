
# include <Siv3D.hpp>


#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <climits>
#include <cfloat>
#include <cstring>
#include <map>
#include <utility>
#include <set>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <list>
#include <algorithm>
#include <functional>
#include <sstream>
#include <complex>
#include <stack>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <array>
#include <cassert>
#include <bitset>
#include <random>
#include <fstream>
using namespace std;
using LL = long long;
using LD = long double;

LD sigmoid(LD x)
{
	return 1. / (1. + exp(-x));
}

LD sigmoid_grad(LD x)
{
	return (1. - sigmoid(x)) * sigmoid(x);
}

template
<typename T>
struct matrix
{
	vector<vector<T>> dat;
	int A, B;
	matrix() :A(0), B(0)
	{}
	matrix(size_t a, size_t b) :
		A(a), B(b)
	{
		dat.resize(A, vector<T>(B));
	}
	T& v(int i, int j)
	{
		assert(i >= 0 && j >= 0 && i < A && j < B);
		return dat[i][j];
	}
	const T& v(int i, int j) const
	{
		assert(i >= 0 && j >= 0 && i < A && j < B);
		return dat[i][j];
	}
	matrix<T> operator*(const matrix& o) const
	{
		matrix<T> res(A, o.B);
		assert(B == o.A);
		for (int i = 0; i < A; ++i)
		{
			for (int j = 0; j < o.B; ++j)
			{
				for (int k = 0; k < B; ++k)
				{
					res.v(i, j) += v(i, k) * o.v(k, j);
				}
			}
		}
		return res;
	}
	matrix<T> operator*(T K) const
	{
		matrix<T> res(A, B);
		for (int i = 0; i < A; ++i)
		{
			for (int j = 0; j < B; ++j)
			{
				res.v(i, j) = v(i, j) * K;
			}
		}
		return res;
	}
	matrix<T> operator+(const matrix& o) const
	{
		matrix<T> res(A, B);
		assert(A == o.A && B == o.B);
		for (int i = 0; i < A; ++i)
		{
			for (int j = 0; j < B; ++j)
			{
				res.v(i, j) = v(i, j) + o.v(i, j);
			}
		}
		return res;
	}
	matrix<T> operator-(const matrix& o) const
	{
		matrix<T> res(A, B);
		assert(A == o.A && B == o.B);
		for (int i = 0; i < A; ++i)
		{
			for (int j = 0; j < B; ++j)
			{
				res.v(i, j) = v(i, j) - o.v(i, j);
			}
		}
		return res;
	}
	matrix<T> transpose() const
	{
		matrix<T> res(B, A);
		for (int i = 0; i < A; ++i)
		{
			for (int j = 0; j < B; ++j)
			{
				res.v(j, i) = v(i, j);
			}
		}
		return res;
	}
	void sig()
	{
		for (int i = 0; i < A; ++i)
		{
			for (int j = 0; j < B; ++j)
			{
				v(i, j) = sigmoid(v(i, j));
			}
		}
	}
	void sig_grad()
	{
		for (int i = 0; i < A; ++i)
		{
			for (int j = 0; j < B; ++j)
			{
				v(i, j) = sigmoid_grad(v(i, j));
			}
		}
	}
	void sof()
	{
		for (int i = 0; i < A; ++i)
		{
			T sum(0);
			for (int j = 0; j < B; ++j)
			{
				sum += exp(v(i, j));
			}
			for (int j = 0; j < B; ++j)
			{
				v(i, j) = exp(v(i, j)) / sum;
			}
		}
	}
	void output()
	{
		for (int i = 0; i < A; ++i)
		{
			for (int j = 0; j < B; ++j)
			{
				cout << dat[i][j] << " \n"[j == B - 1];
			}
		}
	}
};

matrix<LD> mult(matrix<LD> P, const matrix<LD>& Q)
{
	assert(P.A == Q.A && P.B && Q.B);
	for (int i = 0; i < P.A; ++i)
		for (int j = 0; j < P.B; ++j)
			P.v(i, j) = P.v(i, j) * Q.v(i, j);
	return P;
}


LD cross_entropy_error(const matrix<LD>& y, matrix<LD> t)
{
	int batchnum = y.A;
	matrix<LD>ind(t.A, 1);
	if (t.A == y.A)
	{
		for (int i = 0; i < t.A; ++i)
		{
			for (int j = 0; j < t.B; ++j)
			{
				assert(t.v(i, j) == 0. || t.v(i, j) == 1.);
				if (t.v(i, j) > 0.5)//1だった場合
				{
					ind.v(i, 0) = j;
				}
			}
		}
	}
	t = ind;
	LD res = 0;
	for (int i = 0; i < t.A; ++i)
		res += log(y.v(i, t.v(i, 0)));
	res /= -batchnum;
	return res;
}

//引数xをとる関数fについて、各xの要素に対して偏微分を行い、その結果を行列形式で返す
matrix<LD> numerical_gradient_f(function<LD(matrix<LD>)>f, matrix<LD>x)
{
	x = x.transpose();
	LD h = 1e-4;
	matrix<LD>grad(x.A, x.B);
	for (int i = 0; i < x.A; ++i)
	{
		for (int j = 0; j < x.B; ++j)
		{
			auto tmpval = x.v(i, j);
			x.v(i, j) = tmpval + h;
			auto fxh1 = f(x);
			x.v(i, j) = tmpval - h;
			auto fxh2 = f(x);

			grad.v(i, j) = (fxh1 - fxh2) / (2 * h);

			x.v(i, j) = tmpval;
		}
	}
	return grad;
}

#include <ctime>

class TwoLayerNet
{
public:
	matrix<LD> W1, b1, W2, b2;
public:
	TwoLayerNet(int insize, int hidsize, int outsize, LD weightinit = 0.01) :
		W1(insize, hidsize),
		b1(hidsize, 1),
		W2(hidsize, outsize),
		b2(outsize, 1)
	{
		srand(time(NULL));
		for (int i = 0; i < insize; ++i)
		{
			for (int j = 0; j < hidsize; ++j)
			{
				W1.v(i, j) = weightinit * ((LD)rand() / RAND_MAX);
			}
		}
		for (int j = 0; j < hidsize; ++j)
		{
			b1.v(j, 0) = 0.;
		}
		for (int i = 0; i < hidsize; ++i)
		{
			for (int j = 0; j < outsize; ++j)
			{
				W2.v(i, j) = weightinit * ((LD)rand() / RAND_MAX);
			}
		}
		for (int j = 0; j < outsize; ++j)
		{
			b2.v(j, 0) = 0.;
		}
	}
	//xからy(x;W1 b1 W2 b2)を生成する
	matrix<LD> predict(const matrix<LD>& x) const
	{
		int batchnum = x.A;

		matrix<LD> B1(batchnum, W1.B);
		for (int i = 0; i < batchnum; ++i)
			for (int j = 0; j < W1.B; ++j)
				B1.v(i, j) = b1.v(j, 0);

		matrix<LD> B2(batchnum, W2.B);
		for (int i = 0; i < batchnum; ++i)
			for (int j = 0; j < W2.B; ++j)
				B2.v(i, j) = b2.v(j, 0);

		auto a1 = x * W1 + B1;
		a1.sig();
		auto a2 = a1 * W2 + B2;
		a2.sof();
		return a2;
	}
	//入力xの交差エントロピー誤差
	LD loss(const matrix<LD>& x, const matrix<LD>& t)
	{
		auto y = predict(x);
		return cross_entropy_error(y, t);
	}
	//どのくらいの確率で成功したか
	LD accuracy(const matrix<LD>& x, const matrix<LD>& t)
	{
		auto y = predict(x);
		matrix<LD>yarg(y.A, 1);
		for (int i = 0; i < y.A; ++i)
		{
			assert(y.B);
			int bigind = 0;
			LD big = y.v(i, 0);
			for (int j = 0; j < y.B; ++j)
			{
				if (big < y.v(i, j))
				{
					big = y.v(i, j);
					bigind = j;
				}
			}
			yarg.v(i, 0) = bigind;
		}
		int cnt = 0;
		for (int i = 0; i < t.A; ++i)
		{
			assert(t.B);
			int bigind = 0;
			LD big = t.v(i, 0);
			for (int j = 0; j < t.B; ++j)
			{
				if (big < t.v(i, j))
				{
					big = t.v(i, j);
					bigind = j;
				}
			}
			if (bigind == yarg.v(i, 0))
			{
				++cnt;
			}
		}
		LD acc = (LD)cnt / x.A;
		return acc;
	}
	map<string, matrix<LD>> numerical_gradient(const matrix<LD>& x, const matrix<LD>& t)
	{
		function<LD(matrix<LD>)>loss_W = [&](matrix<LD>x_) {return loss(x_, t); };
		map<string, matrix<LD>>grads;
		grads["W1"] = numerical_gradient_f(loss_W, W1);
		grads["b1"] = numerical_gradient_f(loss_W, b1);
		grads["W2"] = numerical_gradient_f(loss_W, W2);
		grads["b2"] = numerical_gradient_f(loss_W, b2);
		return grads;
	}
	map<string, matrix<LD>> gradient(const matrix<LD>& x, const matrix<LD>& t)
	{
		map<string, matrix<LD>>grads;
		int batchnum = x.A;
		//前に計算していく
		matrix<LD> B1(batchnum, W1.B);
		for (int i = 0; i < batchnum; ++i)
			for (int j = 0; j < W1.B; ++j)
				B1.v(i, j) = b1.v(j, 0);
		auto z1 = x * W1 + B1;
		auto a1 = z1;
		z1.sig();
		matrix<LD> B2(batchnum, W2.B);
		for (int i = 0; i < batchnum; ++i)
			for (int j = 0; j < W2.B; ++j)
				B2.v(i, j) = b2.v(j, 0);
		auto y = z1 * W2 + B2;
		auto a2 = y;
		y.sof();
		//逆から計算する
		auto dy = (y - t) * (1. / batchnum);
		grads["W2"] = z1.transpose() * dy;
		grads["b2"] = matrix<LD>(dy.B, 1);
		for (int i = 0; i < dy.A; ++i)
		{
			for (int j = 0; j < dy.B; ++j)
			{
				grads["b2"].v(j, 0) += dy.v(i, j);
			}
		}

		auto da1 = dy * W2.transpose();
		a1.sig_grad();
		auto dz1 = mult(a1, da1);
		grads["W1"] = x.transpose() * dz1;
		grads["b1"] = matrix<LD>(dz1.B, 1);
		for (int i = 0; i < dz1.A; ++i)
		{
			for (int j = 0; j < dz1.B; ++j)
			{
				grads["b1"].v(j, 0) += dz1.v(i, j);
			}
		}

		return grads;
	}
};

const int inputsize = 100;
const int hiddensize = 1000;
const int outputsize = 2;
const int itersnum = 100;//繰り返す回数
int trainsize;
const int batchsize = 1000;//バッチの大きさ(100)
const LD learning_rate = 0.4;//学習係数
vector<vector<int>>vec;
vector<int>ans;

void Main()
{
	Window::Resize(400, 400);
	int arr[10][10] = {};

	Graphics::SetBackground(Palette::White);
	TextWriter tx(L"in.txt");
	int key = 0;
	const Font font(18), font2(24);
	int cnt = 0;
	ifstream ifs("matrix.txt");
	TwoLayerNet network(inputsize, hiddensize, outputsize, -0.01);

	for (int i = 0; i < inputsize; ++i)
		for (int j = 0; j < hiddensize; ++j)
			ifs >> network.W1.v(i, j);
	for (int i = 0; i < hiddensize; ++i)
		for (int j = 0; j < 1; ++j)
			ifs >> network.b1.v(i, j);
	for (int i = 0; i < hiddensize; ++i)
		for (int j = 0; j < outputsize; ++j)
			ifs >> network.W2.v(i, j);
	for (int i = 0; i < outputsize; ++i)
		for (int j = 0; j < 1; ++j)
			ifs >> network.b2.v(i, j);
	ifs.close();
	double left = 0.5, right = 0.5;
	int loop = 0;

	while (System::Update())
	{
		//処理
		auto p = Mouse::Pos();
		int x = p.x / 40, y = p.y / 40;
		x = Clamp(x, 0, 9);
		y = Clamp(y, 0, 9);

		if (Input::MouseL.pressed)
		{
			arr[x][y] = true;
		}
		if (Input::MouseR.pressed)
		{
			arr[x][y] = false;
		}
		if (Input::KeyEnter.clicked)
		{
			++cnt;
			for (int xx = 0; xx < 10; ++xx)
			{
				for (int yy = 0; yy < 10; ++yy)
				{
					tx.write(arr[xx][yy]);
					arr[xx][yy] = 0;
				}
			}
			tx.writeln(Format(L" ", key));
		}
		if (Input::KeySpace.clicked)
		{
			key++;
			key %= 10;
		}
		if (loop % 60 == 0)
		{
			matrix<LD> X(1, inputsize);
			for (int i = 0; i < 10; ++i)
			{
				for (int j = 0; j < 10; ++j)
				{
					X.v(0, i * 10 + j) = arr[i][j];
				}
			}
			auto res = network.predict(X);
			left = res.v(0, 0);
			right = res.v(0, 1);
		}
		if (Input::KeyC.clicked)
		{
			for (int i = 0; i < 10; ++i)
			{
				for (int j = 0; j < 10; ++j)
				{
					arr[i][j] = 0;
				}
			}
		}
		++loop;
		//描画
		Rect(x * 40, y * 40, 40, 40).draw(Palette::Gray);
		for (int xx = 0; xx < 10; ++xx)
		{
			for (int yy = 0; yy < 10; ++yy)
			{
				if (arr[xx][yy])
				{
					Rect(xx * 40, yy * 40, 40, 40).draw(Palette::Black);
				}
			}
		}
		String u = Format(key);
		font.draw(u, { 0,0 }, Palette::Black);
		String s = Format(cnt);
		font.draw(s, { 0,360 }, Palette::Black);
		
		String t = Format(L"Circle:", right, L"\n Cross:", left);
		font2.drawCenter(t, { 200,200 }, Palette::Red);
		
	}
	tx.close();
}
