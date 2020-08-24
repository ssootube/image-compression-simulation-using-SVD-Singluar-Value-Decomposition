#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>
#include <iostream>
#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
using namespace sf;
using namespace std;
using namespace Eigen;
typedef Eigen::Matrix< float, Eigen::Dynamic, Eigen::Dynamic > Mat;

void reset(Mat& mat) {
	//initialize matrix to zero matrix
	//행렬을 영행렬로 초기화합니다.
	for (int i = 0; i < mat.rows(); ++i)
		for (int j = 0; j < mat.cols(); ++j)
			mat(i, j) = 0;
}

bool compare(pair<double, Mat> a, pair<double, Mat> b) {
	//compare function for sorting singular values in descending order
	//특잇값을 내림차순으로 정렬하기 위한 함수
	//제곱근 연산을 절약하기 위해 (A^T)*(A)의 고유값의 제곱근이 아니라, 고유값 자체를 비교합니다.
	//(A^T)*(A)의 고유값은 항상 양수 범위 내에 있으므로 제곱하여 비교하지 않아도 됩니다.
	return a.first > b.first;
}

class RGB {
public:
	enum { R, G, B };
	Mat output[3];
	Mat origin[3];
	int x_size;
	int y_size;
	bool empty = true;
	vector<pair<double, Mat>> lists[3];
	bool processed[3] = { false,false,false };
	int added[3] = { -1,-1,-1 };

	RGB() {
	}

	RGB(Mat red, Mat green, Mat blue) {
		x_size = red.cols();
		y_size = red.rows();
		origin[R] = red; origin[G] = green; origin[B] = blue;
		output[R] = red; output[G] = green; output[B] = blue;
		empty = false;
	}

	void getSVDColor(int color, int rate) {
		if (!processed[color]) {
			Mat ATA = origin[color].transpose() * origin[color]; // (A^T)*(A) 행렬입니다.
			SelfAdjointEigenSolver<Mat> eigensolver(ATA);
			assert(eigensolver.info() == Success);
			Mat eigenValues = eigensolver.eigenvalues();
			Mat eigenVectors = eigensolver.eigenvectors();
			int num = eigenValues.rows();//고유값의 개수

			for (int i = 0; i < num; ++i) {
				pair<double, Mat> element;
				element.first = eigenValues(i);
				element.second = eigenVectors.col(i);
				lists[color].push_back(element);
			}
			sort(lists[color].begin(), lists[color].end(), compare);
			for (int i = 0; i < num; ++i)
				eigenVectors.col(i) = lists[color][i].second;
			processed[color] = true;
			output[color] = Mat(y_size, x_size);
			reset(output[color]);
		}
		if (added[color] <= rate)
			for (int i = added[color] + 1; i < rate; ++i)
				output[color] += (origin[color] * lists[color][i].second) * lists[color][i].second.transpose();
		else
			for (int i = added[color]; i >= rate; --i)
				output[color] -= (origin[color] * lists[color][i].second) * lists[color][i].second.transpose();
		added[color] = rate - 1;
	}
};

RGB getMatrixFromImage(Image& img) {
	int x_size = img.getSize().x;
	int y_size = img.getSize().y;
	Mat red(y_size, x_size);
	Mat green(y_size, x_size);
	Mat blue(y_size, x_size);

	for (int i = 0; i < x_size; ++i) {
		for (int j = 0; j < y_size; ++j) {
			red(j, i) = img.getPixel(i, j).r;
			green(j, i) = img.getPixel(i, j).g;
			blue(j, i) = img.getPixel(i, j).b;
		}
	}
	return RGB(red, green, blue);
}

Image getImageFromRGB(RGB& rgb) {
	Image  result;
	result.create(rgb.x_size, rgb.y_size);
	for (int i = 0; i < rgb.x_size; ++i) {
		for (int j = 0; j < rgb.y_size; ++j) {
			Color color;
			color.r = rgb.output[RGB::R](j, i);
			color.g = rgb.output[RGB::G](j, i);
			color.b = rgb.output[RGB::B](j, i);
			result.setPixel(i, j, color.toInteger() < 0 ? Color(0) : color);
		}
	}
	return result;
}

int main() {
	string input;
	cout << "input image file name including extension>";
	cin >> input;

	Image original;
	original.loadFromFile(input);

	RenderWindow window(VideoMode(original.getSize().x, original.getSize().y), "Math channel Ssootube Singular Value Decomposition");
	RGB mat;
	mat = getMatrixFromImage(original);
	Image applied = getImageFromRGB(mat);
	Texture t; t.loadFromImage(applied);
	Sprite output(t);
	int k = 0;
	while (window.isOpen()) {
		Event e;
		while (window.pollEvent(e)) {
			if (e.type == Event::Closed)
				window.close();
			if (e.type == Event::KeyPressed)
				if (e.key.code == Keyboard::Enter)
				{
					window.clear();
					window.draw(output);
					window.display();
					if (k <= mat.x_size) {
						mat.getSVDColor(RGB::R, k);
						mat.getSVDColor(RGB::G, k);
						mat.getSVDColor(RGB::B, k++);
						applied = getImageFromRGB(mat);
						t.loadFromImage(applied);
						output.setTexture(t);
						cout << "press enter to continue. current rank:" << ((k - 2 == -1) ? 0 : k - 2) << endl;
					}
					break;
				}
		}
	}
	return 0;
}
