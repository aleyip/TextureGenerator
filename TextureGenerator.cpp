// TextureGenerator.cpp : Este arquivo contém a função 'main'. A execução do programa começa e termina ali.
//

#include <math.h>

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>

#include <cuda.h>
#include <cuda_runtime.h>

#include "defines.h"

#include "Object.h"
#include "Sphere.h"
#include "Cylinder.h"
#include "Cube.h"

#include "RayLight.h"
#include "Texture4D.h"

#include "CudaPointers.h"

#include "Light.h"
#include "PointLight.h"
#include "DirectionalLight.h"

#define NUMOBJ 7
#define NUMLIGHT 1

#define WINDOW_SIZE 1
#define BUFFERSIZE 50000000

#define specPower 0.3
#define specShiny 16
#define ambientPower 0.1

#define TEXU 512
#define TEXV 256
#define TEXS 33
#define TEXT 33

using ObjectT = Object<typeT>;
using SphereT = Sphere<typeT>;
using CylinderT = Cylinder<typeT>;
using CubeT = Cube<typeT>;

using RayLightT = RayLight<typeT>;
using Texture4DT = Texture4D<typeT>;
using vec3T = vec3<typeT>;
using CudaPointersT = CudaPointers<typeT>;

using LightT = Light<typeT>;
using PointLightT = PointLight<typeT>;
using DirectionalLightT = DirectionalLight<typeT>;

void generateViewPort(ObjectT** obj, LightT** light, vec3T viewerPos, vec3T viewerDir, typeT fov = 60) {
	//PointLight<typeT> light = PointLight<typeT>(vec3T(-6, 4.5, 2), cv::Vec<typeT, 3>(1, 1, 1));
	
	
	cv::Mat img = cv::Mat(2048, 2048, CV_64FC4);
	typeT halfHeight = tan(fov * 0.01745329251994329576923690768489 / 2.);

	viewerDir.normalize();
	vec3T right = vec3(-viewerDir.z, 0., viewerDir.x);
	right.normalize();
	vec3T up = vec3(-right.z * viewerDir.y, right.z * viewerDir.x - right.x * viewerDir.z, right.x * viewerDir.y);
	up.normalize();

	printf("Viewer Position: %f, %f, %f\n", viewerPos.x, viewerPos.y, viewerPos.z);
	printf("Viewer Direction: %f, %f, %f\n", viewerDir.x, viewerDir.y, viewerDir.z);
	img.forEach<cv::Vec<typeT, 4>>([&obj, &light, &viewerPos, &viewerDir, img, halfHeight, &right, &up](cv::Vec<typeT, 4>& pixel, const int* pos)->void {
		typeT dV = ((1.0 - 2.0 * pos[0] / typeT(img.rows - 1)) * halfHeight);
		typeT dH = ((1.0 - 2.0 * pos[1] / typeT(img.cols - 1)) * halfHeight);

		vec3T vDaux = vec3T(1, dV, dH);
		vDaux.normalize();
		vec3T vD = vec3T(
			viewerDir.x * vDaux.x + up.x * vDaux.y + right.x * vDaux.z,
			viewerDir.y * vDaux.x + up.y * vDaux.y + right.y * vDaux.z,
			viewerDir.z * vDaux.x + up.z * vDaux.y + right.z * vDaux.z
		);

		RayLightT ray = RayLightT(viewerPos, vD);
		vec3T normal_out, normal, collision_out, collision;
		int objsel = -1;
		typeT dist = -1;
		for (int i = 0; i < NUMOBJ; i++)
		{
			typeT d = obj[i]->CheckCollision(ray, collision_out, normal_out);
			if ((d < dist && d >= 0) || dist == -1) {
				dist = d;
				normal = normal_out;
				collision = collision_out;
				objsel = i;
			}
		}

		if (dist >= 0) {
			pixel = Light<typeT>::setAmbientLight(*obj[objsel], ambientPower);
			for (int i = 0; i < NUMLIGHT; i++)
				pixel += light[i]->lightEffect(*obj[objsel], collision, normal, viewerPos);
		}
		else pixel = cv::Vec4d(0, 0, 0, 0);
	});

	cv::resize(img, img, cv::Size(720, 720));
	img.convertTo(img, CV_8UC4, 255, 0);
	cv::imshow("teste", img);
	cv::imwrite("D:/testeGen.png", img);
	cv::waitKey(0);
}

void generateViewPortCuda(ObjectT** obj, LightT** light, vec3T viewerPos, vec3T viewerDir, typeT fov = 60) {

	cv::Mat img = cv::Mat(2048, 2048, CV_64FC4);
	typeT halfHeight = tan(fov * 0.01745329251994329576923690768489 / 2.);

	viewerDir.normalize();
	vec3T right = vec3(-viewerDir.z, 0., viewerDir.x);
	right.normalize();
	vec3T up = vec3(-right.z * viewerDir.y, right.z * viewerDir.x - right.x * viewerDir.z, right.x * viewerDir.y);
	up.normalize();

	printf("Viewer Position: %f, %f, %f\n", viewerPos.x, viewerPos.y, viewerPos.z);
	printf("Viewer Direction: %f, %f, %f\n", viewerDir.x, viewerDir.y, viewerDir.z);

	CudaPointersT cp;
	cp.allocate(img.total(), 0, 0, 0, 0);
	{
		std::vector <cv::Vec<typeT, 4>> objColorList(NUMOBJ);
		std::vector <uint8_t> objShinyList(NUMOBJ);
		for (int i = 0; i < NUMOBJ; i++) {
			objColorList[i] = obj[i]->color;
			objShinyList[i] = obj[i]->specularShininness;
		}
		cp.uploadObjectColorProp(objColorList, objShinyList);
	}
	cp.setHitObjectList(-1, img.total());
	cp.setDistList(10e10, img.total());

	std::vector<RayLightT> rayList = std::vector<RayLightT>(img.total());
	for (int r = 0; r < img.rows; r++)
		for (int c = 0; c < img.cols; c++)
		{
			typeT dV = ((1.0 - 2.0 * r / typeT(img.rows - 1)) * halfHeight);
			typeT dH = ((1.0 - 2.0 * c / typeT(img.cols - 1)) * halfHeight);

			vec3T vDaux = vec3T(1, dV, dH);
			vDaux.normalize();
			vec3T vD = vec3T(
				viewerDir.x * vDaux.x + up.x * vDaux.y + right.x * vDaux.z,
				viewerDir.y * vDaux.x + up.y * vDaux.y + right.y * vDaux.z,
				viewerDir.z * vDaux.x + up.z * vDaux.y + right.z * vDaux.z
			);

			rayList[img.cols * r + c] = RayLightT(viewerPos, vD);
		}
	cp.uploadRayList(rayList);

	for (int j = 0; j < NUMOBJ; j++)
		obj[j]->CheckCollisionCuda(cp, img.total(), j);

	Light<typeT>::setAmbientLightCUDA(cp, ambientPower, img.total());
	for (int j = 0; j < NUMLIGHT; j++)
		light[j]->addLightEffectsCUDA(cp, img.total());

	cp.downloadPixelColor((cv::Vec<typeT, 4> *)(img.data), img.total());
	cp.free();

	cv::resize(img, img, cv::Size(720, 720));
	img.convertTo(img, CV_8UC4, 255, 0);
	cv::imshow("teste2", img);
	cv::imwrite("D:/testeGenCuda.png", img);
	cv::waitKey(0);
}

void generateTexture(ObjectT** obj, LightT** light) {
	const int ws = WINDOW_SIZE * 2 + 1;
	const int wsTotal = ws * ws * ws * ws;

	int size[] = { TEXU,TEXV,TEXS,TEXT };
	Texture4DT tex = Texture4DT(size[0], size[1], size[2], size[3]);

	double inc[] = { M_PI * 2. / size[0], M_PI / (size[1] - 1.), M_PI / (size[2] - 1.), M_PI / (size[3] - 1.) };
	double halfinc[4];
	if (WINDOW_SIZE == 0) {
		halfinc[0] = 0;
		halfinc[1] = 0;
		halfinc[2] = 0;
		halfinc[3] = 0;
	}
	else {
		halfinc[0] = inc[0] / (2. * WINDOW_SIZE);
		halfinc[1] = inc[1] / (2. * WINDOW_SIZE);
		halfinc[2] = inc[2] / (2. * WINDOW_SIZE);
		halfinc[3] = inc[3] / (2. * WINDOW_SIZE);
	}

	typeT radius = 3.5;

	typeT denom = 1. / wsTotal;
	std::cout << denom << std::endl;

	std::cout << "Compile Texture" << std::endl;
	auto start = std::chrono::steady_clock::now();
	for (int t = 0; t < size[3]; t++) {
		auto start = std::chrono::steady_clock::now();
		std::cout << "Step t" << t << std::endl;
		for (int s = 0; s < size[2]; s++) {
			std::cout << "Step t" << t << "/" << size[2] << " s " << s << "/" << size[3] << std::endl;
			auto start = std::chrono::steady_clock::now();
#pragma omp parallel for collapse(2)
			for (int v = 0; v < size[1]; v++) {
				for (int u = 0; u < size[0]; u++)
				{
					double angleBase[] = { -(inc[0] * u - M_PI), inc[1] * v - M_PI_2, inc[2] * s - M_PI_2, inc[3] * t - M_PI_2 };

					cv::Vec<typeT, 4> sumPixel = cv::Vec<typeT, 4>(0, 0, 0, 0);
					cv::Vec<typeT, 4> color;
					vec3T normal_out, normal, collision_out, collision;
					int objsel;
					typeT dist;
					double d;

					//#pragma omp parallel for collapse(4)
					for (int n = -WINDOW_SIZE; n <= WINDOW_SIZE; n++)
						for (int m = -WINDOW_SIZE; m <= WINDOW_SIZE; m++)
							for (int l = -WINDOW_SIZE; l <= WINDOW_SIZE; l++)
								for (int k = -WINDOW_SIZE; k <= WINDOW_SIZE; k++)
								{
									typeT angle[] = { angleBase[0] + n * halfinc[0], angleBase[1] + m * halfinc[1], angleBase[2] + k * halfinc[2], angleBase[3] + l * halfinc[3] };

									vec3T pos = vec3T(radius * sin(angle[0]) * cos(angle[1]), radius * sin(angle[1]), radius * cos(angle[0]) * cos(angle[1]));
									vec3T dir = vec3T(sin(angle[2]) * cos(angle[3]), sin(angle[3]), cos(angle[2]) * cos(angle[3]));

									vec3T versorForward(pos);
									versorForward.normalize();
									vec3T versorRight = vec3T(-versorForward.z, 0., versorForward.x);
									versorRight.normalize();
									vec3T versorUp = vec3T(-versorRight.z * versorForward.y, versorRight.z * versorForward.x - versorRight.x * versorForward.z, versorRight.x * versorForward.y);
									versorUp.normalize();

									vec3T vD = vec3T(
										versorRight.x * dir.x + versorUp.x * dir.y + versorForward.x * dir.z,
										versorRight.y * dir.x + versorUp.y * dir.y + versorForward.y * dir.z,
										versorRight.z * dir.x + versorUp.z * dir.y + versorForward.z * dir.z
									);
									vD = vD * (-1);
									RayLightT ray = RayLightT(pos, vD);
									vec3T collision, normal;

									//printf("%d %d %d %d %d %d %d %d angle: %f %f %f %f pos: %f %f %f dir: %f %f %f ", u, v, s, t, k, l, m, n, angle[0], angle[1], angle[2], angle[3], ray.origin.x, ray.origin.y, ray.origin.z, ray.direction.x, ray.direction.y, ray.direction.z);

									color = cv::Vec<typeT, 4>(0, 0, 0, 0);
									objsel = -1;
									dist = 10e10;
									for (int i = 0; i < NUMOBJ; i++)
									{
										d = obj[i]->CheckCollision(ray, collision, normal);
										if (d < dist && d >= 0) {
											dist = d;
											normal = normal_out;
											collision = collision_out;
											objsel = i;
										}
									}
									if (objsel != -1) {
										color = LightT::setAmbientLight(*obj[objsel], ambientPower);
										for (int i = 0; i < NUMLIGHT; i++)
											color += light[i]->lightEffect(*obj[objsel], collision, normal, ray.origin);
									}
									sumPixel += color;
								}
					tex(u, v, s, t) = sumPixel * denom;
				}
			}
			auto end = std::chrono::steady_clock::now();
			std::cout << "Elapsed time in milliseconds: "
				<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
				<< " ms" << std::endl;
		}
		auto end = std::chrono::steady_clock::now();
		std::cout << "Elapsed time in milliseconds: "
			<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
			<< " ms" << std::endl;
	}

	cv::threshold(tex.texture, tex.texture, 1, 1, cv::THRESH_TRUNC);

	cv::Mat img = cv::Mat(256, 512, CV_64FC4);
	for (int u = 0; u < 512; u++)
		for (int v = 0; v < 256; v++)
			img.at<cv::Vec4d>(v, u) = tex(u, v, 16, 16);
	img.convertTo(img, CV_8UC4, 255);
	cv::imshow("Teste", img);
	cv::imwrite("D:/testeTexGen.png", img);
	cv::waitKey(0);

	{
		auto start = std::chrono::steady_clock::now();
		tex.compileToUnity("D:\\testeGenerator.asset");
		auto end = std::chrono::steady_clock::now();
		std::cout << "Elapsed time in milliseconds: "
			<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
			<< " ms" << std::endl;
	}
}

void generateTextureCuda(ObjectT** obj, LightT** light) {
	const int ws = WINDOW_SIZE * 2 + 1;
	const int wsTotal = ws * ws * ws * ws;

	const int bufferSize = BUFFERSIZE / wsTotal;
	const int totalSize = TEXU * TEXV * TEXS * TEXT;

	int size[] = { TEXU,TEXV,TEXS,TEXT };
	Texture4DT tex = Texture4DT(size[0], size[1], size[2], size[3]);

	double inc[] = { M_PI * 2. / size[0], M_PI / (size[1] - 1.), M_PI / (size[2] - 1.), M_PI / (size[3] - 1.) };

	typeT radius = 3.5;

	typeT denom = 1. / wsTotal;
	std::cout << "Denominador: " << denom << std::endl;

	CudaPointersT cp;
	cp.allocate(bufferSize * wsTotal, TEXU, TEXV, TEXS, TEXT);
	{
		std::vector <cv::Vec<typeT, 4>> objColorList(NUMOBJ);
		std::vector <uint8_t> objShinyList(NUMOBJ);
		for (int i = 0; i < NUMOBJ; i++) {
			objColorList[i] = obj[i]->color;
			objShinyList[i] = obj[i]->specularShininness;
		}
		cp.uploadObjectColorProp(objColorList, objShinyList);
	}
	std::cout << "Allocate: " << bufferSize * wsTotal << std::endl;

	std::cout << "Compile Texture" << std::endl;
	int countLoop = 0;//size[3] * (size[2] * (size[1] * 0 + 0) + 16) + 16;
	int length = bufferSize;
	int bufferTotal;
	cv::Vec<typeT, 4>* dataPixel = (cv::Vec<typeT, 4>*)tex.texture.data;

	while (countLoop < totalSize)
	{
		auto start = std::chrono::steady_clock::now();

		bufferTotal = length * wsTotal;

		std::cout << countLoop << "/" << totalSize << std::endl;

		tex.RayLightGeneratorCuda(countLoop, length, radius, ws, wsTotal, cp);

		cp.setHitObjectList(-1, bufferTotal);
		cp.setDistList(10e10, bufferTotal);
		for (int o = 0; o < NUMOBJ; o++)
		{
			obj[o]->CheckCollisionCuda(cp, bufferTotal, o);
		}
		LightT::setAmbientLightCUDA(cp, 0.1, bufferTotal);
		for (int j = 0; j < NUMLIGHT; j++)
			light[j]->addLightEffectsCUDA(cp, bufferTotal);

		cp.pixelReduction(wsTotal, length);
		cp.downloadPixelColor(dataPixel, length);

		auto end = std::chrono::steady_clock::now();
		std::cout << " Compare Elapsed time in milliseconds: "
			<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
			<< " ms" << std::endl;
		//return;
		dataPixel += length;
		countLoop += length;
		if (totalSize - countLoop < bufferSize)
			length = totalSize - countLoop;
	} 
	//delete& dist;
	//delete& objSel;
	//delete& d;

	cp.free();

	cv::Mat img = cv::Mat(TEXV, TEXU, CV_64FC4);
	for (int u = 0; u < TEXU; u++)
		for (int v = 0; v < TEXV; v++)
			//img.at<cv::Vec4d>(v, u) = tex(u, v, 0, 0);
			img.at<cv::Vec4d>(v, u) = tex(u, v, TEXS / 2, TEXT / 2);
	img.convertTo(img, CV_8UC4, 255);
	cv::imwrite("D:/testeGen.png", img);
	cv::imshow("Teste", img);
	cv::imwrite("D:/testeTexGenCuda.png", img);
	cv::waitKey(0);

	{
		auto start = std::chrono::steady_clock::now();
		tex.compileToUnity("D:\\testeGeneratorCuda.asset");
		auto end = std::chrono::steady_clock::now();
		std::cout << "Elapsed time in milliseconds: "
			<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
			<< " ms" << std::endl;
	}
}

int main()
{
	ObjectT* obj[NUMOBJ];
	obj[0] = new SphereT(vec3T(0, 0, 0), 1, cv::Vec<typeT, 3>(1, 1, 1), specPower, specShiny);
	obj[1] = new SphereT(vec3T(0, 0, -1.87), 1, cv::Vec<typeT, 3>(0, 0, 1), specPower, specShiny);
	obj[2] = new CylinderT(vec3T(2.62, 0, 0), vec3T(0, 0, 0), 1, 2, cv::Vec<typeT, 3>(0.941, 0.125, 0.627), specPower, specShiny);
	obj[3] = new CubeT(vec3T(-1.2, .71, -.68), vec3T(0, 0, 0), 1, 1, 1, cv::Vec<typeT, 3>(0, 1, 0), specPower, specShiny);
	obj[4] = new CubeT(vec3T(1.47, -1.05, .93), vec3T(0, 0, 0), 1, 1, 1, cv::Vec<typeT, 3>(0, 1, 1), specPower, specShiny);
	obj[5] = new CylinderT(vec3T(-1.07, -.41, 1.81), vec3T(90, 0, 0), 1, 2, cv::Vec<typeT, 3>(1, 0, 0), specPower, specShiny);
	obj[6] = new CylinderT(vec3T(.87, .76, 2.06), vec3T(0, 0, 0), 1, 2, cv::Vec<typeT, 3>(0, 0, 0), specPower, specShiny);

	LightT* light[NUMLIGHT];
	light[0] = new DirectionalLightT(vec3T(-6, 0, 0), vec3T(1, 1, 1));

	//generateTexture(obj, light);
	generateTextureCuda(obj, light);
	//generateViewPort(obj, light, vec3T(-6, 0, 0), vec3T(1, 0, 0), 90);
	//generateViewPortCuda(obj, light, vec3T(-6, 0, 0), vec3T(1, 0, 0), 90);

	return 0;
}

// Executar programa: Ctrl + F5 ou Menu Depurar > Iniciar Sem Depuração
// Depurar programa: F5 ou menu Depurar > Iniciar Depuração

// Dicas para Começar: 
//   1. Use a janela do Gerenciador de Soluções para adicionar/gerenciar arquivos
//   2. Use a janela do Team Explorer para conectar-se ao controle do código-fonte
//   3. Use a janela de Saída para ver mensagens de saída do build e outras mensagens
//   4. Use a janela Lista de Erros para exibir erros
//   5. Ir Para o Projeto > Adicionar Novo Item para criar novos arquivos de código, ou Projeto > Adicionar Item Existente para adicionar arquivos de código existentes ao projeto
//   6. No futuro, para abrir este projeto novamente, vá para Arquivo > Abrir > Projeto e selecione o arquivo. sln
