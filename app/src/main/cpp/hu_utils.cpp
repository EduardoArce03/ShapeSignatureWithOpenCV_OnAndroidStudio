// Created by eduardo on 28/6/25.
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "hu_utils.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <android/log.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <iomanip>

std::vector<HuDescriptor> cargarCSV(const std::string& ruta_csv) {
    std::vector<HuDescriptor> descriptores;
    std::ifstream file(ruta_csv);

    if (!file.is_open()) {
        __android_log_print(ANDROID_LOG_ERROR, "HU", "No se pudo abrir archivo: %s", ruta_csv.c_str());
        return descriptores;
    }

    std::string linea;
    std::getline(file, linea); // ⚠️ Saltar encabezado: label, hu1,..., fft64

    while (std::getline(file, linea)) {
        std::stringstream ss(linea);
        std::string celda;
        HuDescriptor desc;

        if (!std::getline(ss, celda, ',')) continue;
        desc.label = celda;

        for (int i = 0; i < 7; ++i) {
            if (!std::getline(ss, celda, ',')) break;
            try {
                desc.hu.push_back(std::stod(celda));
            } catch (...) {
                desc.hu.push_back(0.0);
            }
        }

        for (int i = 0; i < 64; ++i) {
            if (!std::getline(ss, celda, ',')) break;
            try {
                desc.fft.push_back(std::stod(celda));
            } catch (...) {
                desc.fft.push_back(0.0);
            }
        }

        if (desc.hu.size() == 7 && desc.fft.size() == 64) {
            descriptores.push_back(desc);
        } else {
            __android_log_print(ANDROID_LOG_WARN, "HU", "Fila inválida: %s (hu=%zu, fft=%zu)",
                                desc.label.c_str(), desc.hu.size(), desc.fft.size());
        }
    }

    __android_log_print(ANDROID_LOG_INFO, "HU", "CSV cargado: %zu entradas válidas", descriptores.size());
    return descriptores;
}


std::vector<HuDescriptor> cargarCSVDesdeAssets(AAssetManager* assetManager, const std::string& assetPath) {
    std::vector<HuDescriptor> descriptores;

    AAsset* asset = AAssetManager_open(assetManager, assetPath.c_str(), AASSET_MODE_STREAMING);
    if (!asset) {
        __android_log_print(ANDROID_LOG_ERROR, "HU", "No se pudo abrir asset: %s", assetPath.c_str());
        return descriptores;
    }

    off_t length = AAsset_getLength(asset);
    std::string contenido(length, '\0');
    AAsset_read(asset, contenido.data(), length);
    AAsset_close(asset);

    std::istringstream ss(contenido);
    std::string linea;
    std::getline(ss, linea); // Saltar encabezado

    while (std::getline(ss, linea)) {
        std::stringstream lineStream(linea);
        std::string celda;
        HuDescriptor desc;

        if (!std::getline(lineStream, celda, ',')) continue;
        desc.label = celda;

        for (int i = 0; i < 7; ++i) {
            if (!std::getline(lineStream, celda, ',')) break;
            try { desc.hu.push_back(std::stod(celda)); }
            catch (...) { desc.hu.push_back(0.0); }
        }

        for (int i = 0; i < 64; ++i) {
            if (!std::getline(lineStream, celda, ',')) break;
            try { desc.fft.push_back(std::stod(celda)); }
            catch (...) { desc.fft.push_back(0.0); }
        }

        if (desc.hu.size() == 7 && desc.fft.size() == 64) {
            descriptores.push_back(desc);
        } else {
            __android_log_print(ANDROID_LOG_WARN, "HU", "Entrada inválida: %s hu=%zu fft=%zu",
                                desc.label.c_str(), desc.hu.size(), desc.fft.size());
        }
    }

    return descriptores;
}

double distanciaEuclidea(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) return 1e9;
    double suma = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        suma += diff * diff;
    }
    return sqrt(suma);
}

double distanciaCombinada(const std::vector<double>& hu1, const std::vector<double>& fft1,
                          const std::vector<double>& hu2, const std::vector<double>& fft2,
                          double peso_hu, double peso_fft) {
    double dist_hu = distanciaEuclidea(hu1, hu2);
    double dist_fft = distanciaEuclidea(fft1, fft2);
    return peso_hu * dist_hu + peso_fft * dist_fft;
}

bool guardarDescriptorCSV(const std::string& ruta, const HuDescriptor& descriptor, bool append) {
    std::ofstream file;
    file.open(ruta, append ? std::ios::app : std::ios::out);
    if (!file.is_open()) {
        __android_log_print(ANDROID_LOG_ERROR, "HU", "No se pudo abrir archivo para guardar: %s", ruta.c_str());
        return false;
    }

    file << descriptor.label;
    for (const double& val : descriptor.hu)
        file << "," << std::setprecision(10) << val;
    for (const double& val : descriptor.fft)
        file << "," << std::setprecision(10) << val;
    file << "\n";

    file.close();
    return true;
}

std::vector<double> calcularFFT(const std::vector<double>& firma) {
    std::vector<double> magnitudes;
    if (firma.empty()) return magnitudes;

    cv::Mat input = cv::Mat(firma).reshape(1, 1);
    input.convertTo(input, CV_32F);

    cv::Mat complexI;
    cv::dft(input, complexI, cv::DFT_COMPLEX_OUTPUT);

    for (int i = 0; i < complexI.cols; ++i) {
        float real = complexI.at<cv::Vec2f>(0, i)[0];
        float imag = complexI.at<cv::Vec2f>(0, i)[1];
        double mag = std::sqrt(real * real + imag * imag);
        magnitudes.push_back(mag);
    }

    double max_val = *std::max_element(magnitudes.begin(), magnitudes.end());
    if (max_val > 0.0) {
        for (double& m : magnitudes)
            m /= max_val;
    }

    return magnitudes;
}

// ✅ Agrega esto en tu función principal JNI luego de calcular fftMag
void completarFFT64(std::vector<double>& fftMag) {
    if (fftMag.size() < 64) {
        while (fftMag.size() < 64)
            fftMag.push_back(0.0);
    } else if (fftMag.size() > 64) {
        fftMag.resize(64);
    }
}

std::vector<double> calcularShapeSignature(std::vector<cv::Point>& contorno) {
    std::vector<double> firma;
    if (contorno.empty()) return firma;

    cv::Moments m = cv::moments(contorno);
    double cx = m.m10 / m.m00;
    double cy = m.m01 / m.m00;

    for (const auto& punto : contorno) {
        double dx = punto.x - cx;
        double dy = punto.y - cy;
        firma.push_back(std::sqrt(dx * dx + dy * dy));
    }

    return firma;
}

std::vector<ZernikeDescriptor> cargarZernikeCSV(const std::string& ruta_csv) {
    std::vector<ZernikeDescriptor> descriptores;
    std::ifstream file(ruta_csv);
    if (!file.is_open()) return descriptores;

    std::string linea;
    std::getline(file, linea); // saltar encabezado

    while (std::getline(file, linea)) {
        std::stringstream ss(linea);
        std::string celda;
        ZernikeDescriptor d;

        if (!std::getline(ss, celda, ',')) continue;
        d.label = celda;

        while (std::getline(ss, celda, ',')) {
            d.zernike.push_back(std::stod(celda));
        }

        descriptores.push_back(d);
    }

    return descriptores;
}

