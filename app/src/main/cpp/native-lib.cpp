#include <jni.h>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <android/bitmap.h>
#include <android/log.h>
#include "hu_utils.h"
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <fstream>  // <- Necesario para std::ofstream
#include <map>
#include <iomanip>

void normalizarVector(std::vector<double>& vec, const std::vector<double>& media, const std::vector<double>& desviacion) {
    for (size_t i = 0; i < vec.size(); ++i) {
        if (i < media.size() && i < desviacion.size() && desviacion[i] != 0) {
            vec[i] = (vec[i] - media[i]) / desviacion[i];
        }
    }
}
ma
extern "C" JNIEXPORT jstring JNICALL
Java_com_example_visio_1p3_MainActivity_stringFromJNI(
        JNIEnv* env,
        jobject /* this */) {

    // Crea una imagen negra 100x100
    cv::Mat img = cv::Mat::zeros(100, 100, CV_8UC3);

    // Dibuja un círculo blanco en el centro
    cv::circle(img, cv::Point(50, 50), 25, cv::Scalar(255, 255, 255), -1);

    // Verifica si el píxel central se pintó correctamente
    cv::Vec3b centro = img.at<cv::Vec3b>(50, 50);
    std::string resultado;

    if (centro == cv::Vec3b(255, 255, 255)) {
        resultado = "OpenCV funcionando ✅ - Círculo dibujado.";
    } else {
        resultado = "OpenCV cargado, pero algo falló ❌.";
    }

    return env->NewStringUTF(resultado.c_str());
}


double radialPolynomial(int n, int m, double r) {
    double R = 0.0;
    m = std::abs(m);
    for (int s = 0; s <= (n - m) / 2; ++s) {
        double num = std::pow(-1, s) * tgamma(n - s + 1);
        double denom = tgamma(s + 1) *
                       tgamma((n + m) / 2 - s + 1) *
                       tgamma((n - m) / 2 - s + 1);
        R += (num / denom) * std::pow(r, n - 2 * s);
    }
    return R;
}
double distanciaEuclidiana(const std::vector<double>& a, const std::vector<double>& b) {
    double suma = 0.0;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        suma += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(suma);
}


std::vector<double> calculateZernikeMoments(cv::Mat &binaryImage) {
    int maxOrder = 4;
    int cx = binaryImage.cols / 2;
    int cy = binaryImage.rows / 2;
    double normFactor = sqrt(cx * cx + cy * cy);
    std::vector<double> moments;

    for (int n = 0; n <= maxOrder; n++) {
        for (int m = -n; m <= n; m += 2) {
            std::complex<double> moment(0.0, 0.0);

            for (int y = 0; y < binaryImage.rows; y++) {
                for (int x = 0; x < binaryImage.cols; x++) {
                    if (binaryImage.at<uchar>(y, x) > 0) {
                        double normX = (x - cx) / normFactor;
                        double normY = (cy - y) / normFactor; // (cy - y)
                        double r = sqrt(normX * normX + normY * normY);
                        double theta = atan2(normY, normX);

                        if (r <= 1.0) {
                            double R = radialPolynomial(n, m, r);
                            std::complex<double> Z(R * cos(m * theta), R * sin(m * theta));
                            moment += Z * (binaryImage.at<uchar>(y, x) / 255.0);
                        }
                    }
                }
            }

            double factor = double(n + 1) / M_PI;
            double value = abs(moment * factor);
            moments.push_back(value);
        }
    }

    return moments;
}


void matToBitmap(JNIEnv* env, const cv::Mat& mat, jobject bitmap) {
    AndroidBitmapInfo info;
    void* pixels = nullptr;

    if (AndroidBitmap_getInfo(env, bitmap, &info) < 0) return;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) return;

    cv::Mat dst(info.height, info.width, CV_8UC4, pixels);

    if (mat.channels() == 1) {
        cv::cvtColor(mat, dst, cv::COLOR_GRAY2RGBA);
    } else if (mat.channels() == 3) {
        cv::cvtColor(mat, dst, cv::COLOR_RGB2RGBA);
    } else {
        mat.copyTo(dst);
    }

    AndroidBitmap_unlockPixels(env, bitmap);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_visio_1p3_MainActivity_procesarYMostrar(JNIEnv *env, jobject thiz,
                                                         jobject inputBitmap, jobject outputBitmap) {
    AndroidBitmapInfo info;
    void* pixels;

    // Obtener info y datos del bitmap
    if (AndroidBitmap_getInfo(env, inputBitmap, &info) < 0 ||
        AndroidBitmap_lockPixels(env, inputBitmap, &pixels) < 0) {
        return;
    }

    cv::Mat src(info.height, info.width, CV_8UC4, pixels);
    AndroidBitmap_unlockPixels(env, inputBitmap);

    // 1. Convertir a escala de grises
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_RGBA2GRAY);

    // 2. Suavizar para evitar ruido
    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);

    // 3. Binarización: mezcla de técnicas
    cv::Mat binGlobal, binAdaptive, bin;
    cv::threshold(gray, binGlobal, 170, 255, cv::THRESH_BINARY);
    cv::adaptiveThreshold(gray, binAdaptive, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                          cv::THRESH_BINARY, 11, 2);

    // 4. Fusionar ambas (AND o combinación ponderada)
    cv::bitwise_and(binGlobal, binAdaptive, bin);

    // 5. Detectar contornos detalladamente
    std::vector<std::vector<cv::Point>> contornos;
    cv::findContours(bin, contornos, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

    // Validar si hay contornos
    if (contornos.empty()) {
        __android_log_print(ANDROID_LOG_WARN, "PROCESAR", "No se encontraron contornos.");
        return;
    }

    // 6. Seleccionar el contorno más grande (filtrar ruido)
    auto max_contorno = *std::max_element(contornos.begin(), contornos.end(),
                                          [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                                              return cv::contourArea(a) < cv::contourArea(b);
                                          });

    double area = cv::contourArea(max_contorno);
    if (area < 300.0) {
        __android_log_print(ANDROID_LOG_WARN, "PROCESAR", "Contorno demasiado pequeño: %.2f", area);
        return;
    }

    // 7. Crear imagen rellena solo del mejor contorno
    cv::Mat filled = cv::Mat::zeros(bin.size(), CV_8UC1);
    cv::drawContours(filled, std::vector<std::vector<cv::Point>>{max_contorno}, -1, cv::Scalar(255), cv::FILLED);

    // 8. Mostrar resultado en outputBitmap
    matToBitmap(env, filled, outputBitmap);
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_visio_1p3_MainActivity_detectarFigura(JNIEnv *env, jobject thiz, jobject bitmap, jobject assetManager) {
    AndroidBitmapInfo info;
    void* pixels;

    if (AndroidBitmap_getInfo(env, bitmap, &info) < 0)
        return env->NewStringUTF("Error obteniendo info de bitmap");

    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0)
        return env->NewStringUTF("Error bloqueando píxeles");

    cv::Mat img(info.height, info.width, CV_8UC4, pixels);
    AndroidBitmap_unlockPixels(env, bitmap);

    cv::Mat gray, bin;
    cv::cvtColor(img, gray, cv::COLOR_RGBA2GRAY);
    cv::threshold(gray, bin, 170, 255, cv::THRESH_BINARY);

    std::vector<std::vector<cv::Point>> contornos;
    cv::findContours(bin, contornos, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contornos.empty())
        return env->NewStringUTF("No se detectaron figuras");

    // Seleccionar el contorno más grande
    auto max_contorno = *std::max_element(contornos.begin(), contornos.end(),
                                          [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                                              return cv::contourArea(a) < cv::contourArea(b);
                                          });

    // Validar área mínima
    double area = cv::contourArea(max_contorno);
    if (area < 100.0)
        return env->NewStringUTF("Figura inválida o muy pequeña");

    // Imagen rellena del contorno más grande
    cv::Mat filled = cv::Mat::zeros(bin.size(), CV_8UC1);
    cv::drawContours(filled, std::vector<std::vector<cv::Point>>{max_contorno}, -1, cv::Scalar(255), cv::FILLED);

    // Momentos de Hu
    std::vector<double> hu(7);
    cv::HuMoments(cv::moments(filled, true), hu.data());
    for (double& h : hu)
        h = -1 * copysign(1.0, h) * log10(std::abs(h));

    // FFT
    std::vector<double> shapeSig = calcularShapeSignature(max_contorno);
    std::vector<double> fftMag = calcularFFT(shapeSig);
    completarFFT64(fftMag);

    // Cargar dataset
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    std::vector<HuDescriptor> dataset = cargarCSVDesdeAssets(mgr, "momentos_hu_dataset.csv");
    if (dataset.empty())
        return env->NewStringUTF("Dataset vacío");

    // Clasificación
    std::string mejorEtiqueta = "Desconocida";
    double mejorDistancia = 1e9;

    for (const auto& entry : dataset) {
        double dist = distanciaCombinada(hu, fftMag, entry.hu, entry.fft, 1.0, 1.0);
        if (dist < mejorDistancia) {
            mejorDistancia = dist;
            mejorEtiqueta = entry.label;
        }
    }

    std::string resultado = "Figura detectada: " + mejorEtiqueta;
    return env->NewStringUTF(resultado.c_str());
}


extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_visio_1p3_MainActivity_guardarDescriptorDesdeApp(
        JNIEnv *env,
        jobject thiz,
        jobject bitmap,
        jstring jlabel,
        jstring jpath) {

    const char* label = env->GetStringUTFChars(jlabel, nullptr);
    const char* path = env->GetStringUTFChars(jpath, nullptr);

    AndroidBitmapInfo info;
    void* pixels;

    if (AndroidBitmap_getInfo(env, bitmap, &info) < 0 ||
        AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) {
        env->ReleaseStringUTFChars(jlabel, label);
        env->ReleaseStringUTFChars(jpath, path);
        return JNI_FALSE;
    }

    cv::Mat img(info.height, info.width, CV_8UC4, pixels);
    cv::Mat gray, bin;
    cv::cvtColor(img, gray, cv::COLOR_RGBA2GRAY);
    cv::threshold(gray, bin, 170, 255, cv::THRESH_BINARY);
    AndroidBitmap_unlockPixels(env, bitmap);

    std::vector<std::vector<cv::Point>> contornos;
    cv::findContours(bin, contornos, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contornos.empty()) {
        __android_log_print(ANDROID_LOG_WARN, "GUARDAR", "No se encontraron contornos");
        env->ReleaseStringUTFChars(jlabel, label);
        env->ReleaseStringUTFChars(jpath, path);
        return JNI_FALSE;
    }

    auto max_contorno = *std::max_element(contornos.begin(), contornos.end(),
                                          [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                                              return cv::contourArea(a) < cv::contourArea(b);
                                          });

    // Validación: área mínima
    double area = cv::contourArea(max_contorno);
    if (area < 500.0) {
        __android_log_print(ANDROID_LOG_WARN, "GUARDAR", "Contorno descartado por área mínima: %.2f", area);
        env->ReleaseStringUTFChars(jlabel, label);
        env->ReleaseStringUTFChars(jpath, path);
        return JNI_FALSE;
    }

    // Imagen rellena solo con el contorno más grande
    cv::Mat contourImg = cv::Mat::zeros(bin.size(), CV_8UC1);
    if (max_contorno.size() > 10) {
        cv::drawContours(contourImg, std::vector<std::vector<cv::Point>>{max_contorno}, -1, cv::Scalar(255), cv::FILLED);
    } else {
        __android_log_print(ANDROID_LOG_WARN, "CONTORNO", "Contorno muy pequeño: %zu puntos", max_contorno.size());
    }
    __android_log_print(ANDROID_LOG_INFO, "AREA", "Área del contorno: %.2f", cv::contourArea(max_contorno));

    std::vector<double> hu(7);
    cv::HuMoments(cv::moments(contourImg, true), hu.data());
    for (double& h : hu)
        h = -1 * copysign(1.0, h) * log10(std::abs(h));

    std::vector<double> shapeSig = calcularShapeSignature(max_contorno);
    std::vector<double> fftMag = calcularFFT(shapeSig);
    completarFFT64(fftMag);

    HuDescriptor descriptor;
    descriptor.label = label;
    descriptor.hu = hu;
    descriptor.fft = fftMag;
    __android_log_print(ANDROID_LOG_INFO, "LABEL", "Figura: %s", label);
    for (int i = 0; i < hu.size(); ++i)
        __android_log_print(ANDROID_LOG_INFO, "HU", "hu[%d] = %.6f", i, hu[i]);
    for (int i = 0; i < 10; ++i)
        __android_log_print(ANDROID_LOG_INFO, "FFT", "fft[%d] = %.6f", i, fftMag[i]);


    std::string ruta(path);
    bool exito = guardarDescriptorCSV(ruta, descriptor, true);

    env->ReleaseStringUTFChars(jlabel, label);
    env->ReleaseStringUTFChars(jpath, path);
    return exito ? JNI_TRUE : JNI_FALSE;
}




extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_visio_1p3_MainActivity_evaluarPrecision(JNIEnv *env, jobject thiz, jobject assetManager) {
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    // Cargar dataset de entrenamiento (desde assets)
    std::vector<HuDescriptor> entrenamiento = cargarCSVDesdeAssets(mgr, "momentos_hu_dataset.csv");
    if (entrenamiento.empty()) {
        return env->NewStringUTF("Dataset de entrenamiento vacío.");
    }

    // Cargar dataset de prueba (desde almacenamiento)
    std::vector<HuDescriptor> prueba = cargarCSV("/sdcard/test_descriptores.csv");
    if (prueba.empty()) {
        return env->NewStringUTF("Dataset de prueba vacío.");
    }

    int aciertos = 0;
    for (const auto& test : prueba) {
        std::string predLabel = "Desconocida";
        double minDist = 1e9;

        for (const auto& train : entrenamiento) {
            double dist = distanciaCombinada(test.hu, test.fft, train.hu, train.fft);
            if (dist < minDist) {
                minDist = dist;
                predLabel = train.label;
            }
        }

        if (predLabel == test.label) {
            aciertos++;
        }
    }

    double precision = 100.0 * aciertos / prueba.size();
    char buffer[64];
    snprintf(buffer, sizeof(buffer), "Precisión: %.2f%% (%d/%zu)", precision, aciertos, prueba.size());

    __android_log_print(ANDROID_LOG_INFO, "EVALUACION", "Resultado: %s", buffer);
    return env->NewStringUTF(buffer);
}
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <sstream>
#include <vector>
#include <string>

std::vector<ZernikeDescriptor> cargarZernikeDesdeAssets(AAssetManager* mgr, const std::string& nombreArchivo) {
    std::vector<ZernikeDescriptor> descriptores;

    AAsset* asset = AAssetManager_open(mgr, nombreArchivo.c_str(), AASSET_MODE_BUFFER);
    if (!asset) {
        __android_log_print(ANDROID_LOG_ERROR, "Zernike", "No se pudo abrir el archivo: %s", nombreArchivo.c_str());
        return descriptores;
    }

    size_t tamano = AAsset_getLength(asset);
    const char* buffer = static_cast<const char*>(AAsset_getBuffer(asset));
    if (!buffer) {
        __android_log_print(ANDROID_LOG_ERROR, "Zernike", "No se pudo leer el archivo");
        AAsset_close(asset);
        return descriptores;
    }

    std::istringstream stream(std::string(buffer, tamano));
    std::string linea;

    // Leer encabezado
    std::getline(stream, linea);

    while (std::getline(stream, linea)) {
        std::stringstream ss(linea);
        std::string celda;

        ZernikeDescriptor desc;

        // Leer etiqueta
        if (!std::getline(ss, celda, ',')) continue;
        desc.label = celda;

        // Leer momentos
        while (std::getline(ss, celda, ',')) {
            try {
                desc.zernike.push_back(std::stod(celda));
            } catch (...) {
                desc.zernike.push_back(0.0);  // si hay error
            }
        }

        descriptores.push_back(desc);
    }

    AAsset_close(asset);
    return descriptores;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_visio_1p3_MainActivity_predecirDesdeApp(JNIEnv *env, jobject thiz, jobject assetManager, jstring metodo) {
    const char *cMetodo = env->GetStringUTFChars(metodo, nullptr);
    std::string metodoStr(cMetodo);
    env->ReleaseStringUTFChars(metodo, cMetodo);

    AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
    std::string resultado;

    if (metodoStr == "Momentos de HU") {
        std::vector<HuDescriptor> entrenamiento = cargarCSVDesdeAssets(mgr, "momentos_hu_dataset.csv");
        std::vector<HuDescriptor> prueba = cargarCSV("/sdcard/test_descriptores.csv");

        if (entrenamiento.empty() || prueba.empty()) {
            return env->NewStringUTF("Dataset HU vacío.");
        }

        // Normalización automática para HU y FFT
        std::vector<double> mean_hu(7, 0.0), std_hu(7, 0.0);
        std::vector<double> mean_fft(64, 0.0), std_fft(64, 0.0);

        for (const auto &e : entrenamiento)
            for (int i = 0; i < 7; ++i) mean_hu[i] += e.hu[i];
        for (const auto &e : entrenamiento)
            for (int i = 0; i < 64; ++i) mean_fft[i] += e.fft[i];
        for (int i = 0; i < 7; ++i) mean_hu[i] /= entrenamiento.size();
        for (int i = 0; i < 64; ++i) mean_fft[i] /= entrenamiento.size();
        for (const auto &e : entrenamiento) {
            for (int i = 0; i < 7; ++i) std_hu[i] += pow(e.hu[i] - mean_hu[i], 2);
            for (int i = 0; i < 64; ++i) std_fft[i] += pow(e.fft[i] - mean_fft[i], 2);
        }
        for (int i = 0; i < 7; ++i) std_hu[i] = sqrt(std_hu[i] / entrenamiento.size());
        for (int i = 0; i < 64; ++i) std_fft[i] = sqrt(std_fft[i] / entrenamiento.size());

        for (auto &e : entrenamiento) {
            normalizarVector(e.hu, mean_hu, std_hu);
            normalizarVector(e.fft, mean_fft, std_fft);
        }
        for (auto &p : prueba) {
            normalizarVector(p.hu, mean_hu, std_hu);
            normalizarVector(p.fft, mean_fft, std_fft);
        }

        int aciertos = 0, total = 0;

        for (const auto &test : prueba) {
            if (test.hu.size() != 7 || test.fft.size() != 64) continue;

            std::string predLabel = "Desconocida";
            double minDist = 1e9;
            std::vector<double> distancias;

            for (const auto &train : entrenamiento) {
                double dist = distanciaCombinada(test.hu, test.fft, train.hu, train.fft);
                distancias.push_back(dist);
                if (dist < minDist) {
                    minDist = dist;
                    predLabel = train.label;
                }
            }

            double maxDist = *std::max_element(distancias.begin(), distancias.end());
            double confianza = 100.0 * (1.0 - (minDist / (maxDist + 1e-8)));
            confianza = std::clamp(confianza, 0.0, 100.0);

            resultado += "Figura detectada: " + predLabel + " (" +
                         std::to_string(confianza) + "%) | Real: " + test.label + "\n";

            if (predLabel == test.label) aciertos++;
            total++;
        }

        double precision = total > 0 ? (100.0 * aciertos / total) : 0.0;
        resultado += "\nPrecisión global: " + std::to_string(precision) + "%";
        return env->NewStringUTF(resultado.c_str());
    }

    else if (metodoStr == "Momentos de Zernike") {
        std::vector<ZernikeDescriptor> entrenamiento = cargarZernikeDesdeAssets(mgr, "momentos_zernike_dataset.csv");
        std::vector<ZernikeDescriptor> prueba = cargarZernikeCSV("/sdcard/test_descriptores.csv");

        if (entrenamiento.empty() || prueba.empty()) {
            return env->NewStringUTF("Dataset Zernike vacío.");
        }

        // Normalización
        size_t dim = entrenamiento[0].zernike.size();
        std::vector<double> mean(dim, 0.0), stddev(dim, 0.0);

        for (const auto &e : entrenamiento)
            for (size_t i = 0; i < dim; ++i)
                mean[i] += e.zernike[i];
        for (auto &m : mean) m /= entrenamiento.size();
        for (const auto &e : entrenamiento)
            for (size_t i = 0; i < dim; ++i)
                stddev[i] += std::pow(e.zernike[i] - mean[i], 2);
        for (auto &s : stddev) s = std::sqrt(s / entrenamiento.size());

        for (auto &e : entrenamiento) normalizarVector(e.zernike, mean, stddev);
        for (auto &e : prueba) normalizarVector(e.zernike, mean, stddev);

        int aciertos = 0, total = 0;

        for (const auto &test : prueba) {
            if (test.zernike.empty()) continue;

            std::string predLabel = "Desconocida";
            double minDist = 1e9;
            std::vector<double> distancias;

            for (const auto &train : entrenamiento) {
                if (train.zernike.size() != test.zernike.size()) continue;

                double dist = distanciaEuclidiana(test.zernike, train.zernike);
                distancias.push_back(dist);
                if (dist < minDist) {
                    minDist = dist;
                    predLabel = train.label;
                }
            }

            double maxDist = *std::max_element(distancias.begin(), distancias.end());
            double confianza = 100.0 * (1.0 - (minDist / (maxDist + 1e-8)));
            confianza = std::clamp(confianza, 0.0, 100.0);

            resultado += "Figura detectada: " + predLabel + " (" +
                         std::to_string(confianza) + "%) | Real: " + test.label + "\n";

            if (predLabel == test.label) aciertos++;
            total++;
        }

        double precision = total > 0 ? (100.0 * aciertos / total) : 0.0;
        resultado += "\nPrecisión global: " + std::to_string(precision) + "%";
        return env->NewStringUTF(resultado.c_str());
    }

    // En caso de que el método no coincida
    return env->NewStringUTF("Método no reconocido.");
}


extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_visio_1p3_MainActivity_clasificarFigura(JNIEnv *env, jobject thiz, jobject bitmap,
                                                         jstring metodo) {
    const char *cMetodo = env->GetStringUTFChars(metodo, nullptr);
    std::string metodoStr(cMetodo);
    env->ReleaseStringUTFChars(metodo, cMetodo);

    AndroidBitmapInfo info;
    void *pixels;

    if (AndroidBitmap_getInfo(env, bitmap, &info) < 0 ||
        AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) {
        return env->NewStringUTF("Error procesando imagen.");
    }

    cv::Mat img(info.height, info.width, CV_8UC4, pixels);
    AndroidBitmap_unlockPixels(env, bitmap);

    cv::Mat gray, bin;
    cv::cvtColor(img, gray, cv::COLOR_RGBA2GRAY);
    cv::threshold(gray, bin, 170, 255, cv::THRESH_BINARY);

    std::vector<std::vector<cv::Point>> contornos;
    cv::findContours(bin, contornos, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contornos.empty()) return env->NewStringUTF("No se detectaron figuras");

    auto max_contorno = *std::max_element(contornos.begin(), contornos.end(),
                                          [](const std::vector<cv::Point> &a,
                                             const std::vector<cv::Point> &b) {
                                              return cv::contourArea(a) < cv::contourArea(b);
                                          });

    if (cv::contourArea(max_contorno) < 100.0) {
        return env->NewStringUTF("Figura muy pequeña o inválida");
    }

    cv::Mat filled = cv::Mat::zeros(bin.size(), CV_8UC1);
    cv::drawContours(filled, std::vector<std::vector<cv::Point>>{max_contorno}, -1, cv::Scalar(255),
                     cv::FILLED);

    std::string resultado;

    if (metodoStr == "Momentos de HU") {
        std::vector<double> hu(7);
        cv::HuMoments(cv::moments(filled, true), hu.data());
        for (double &h: hu)
            h = -1 * copysign(1.0, h) * log10(std::abs(h));

        std::vector<double> shapeSig = calcularShapeSignature(max_contorno);
        std::vector<double> fftMag = calcularFFT(shapeSig);
        completarFFT64(fftMag);

        AAssetManager *mgr = AAssetManager_fromJava(env, env->CallObjectMethod(thiz,
                                                                               env->GetMethodID(
                                                                                       env->GetObjectClass(
                                                                                               thiz),
                                                                                       "getAssets",
                                                                                       "()Landroid/content/res/AssetManager;")));
        std::vector<HuDescriptor> dataset = cargarCSVDesdeAssets(mgr, "momentos_hu_dataset.csv");

        if (dataset.empty()) return env->NewStringUTF("Dataset vacío");

        std::string mejorEtiqueta = "Desconocida";
        double mejorDist = 1e9;

        for (const auto &d: dataset) {
            double dist = distanciaCombinada(hu, fftMag, d.hu, d.fft);
            if (dist < mejorDist) {
                mejorDist = dist;
                mejorEtiqueta = d.label;
            }
        }

        resultado = "Figura detectada: " + mejorEtiqueta;
    } else if (metodoStr == "Momentos de Zernike") {
        std::vector<double> zernike = calculateZernikeMoments(filled);

        std::vector<ZernikeDescriptor> dataset = cargarZernikeCSV(
                "/data/data/com.example.visio_p3/files/momentos_zernike_dataset.csv");
        __android_log_print(ANDROID_LOG_INFO, "ZERNIKE", "Tamaño del dataset Zernike: %zu", dataset.size());

        if (dataset.empty()) return env->NewStringUTF("Dataset Zernike vacío");

        std::string mejorEtiqueta = "Desconocida";
        double mejorDist = 1e9;

        for (const auto &d: dataset) {
            double dist = distanciaEuclidiana(zernike, d.zernike);
            if (dist < mejorDist) {
                mejorDist = dist;
                mejorEtiqueta = d.label;
            }
        }

        resultado = "Figura detectada (Zernike): " + mejorEtiqueta;

    }return env->NewStringUTF(resultado.c_str());

}


bool guardarDescriptorZernikeCSV(const std::string& path, const ZernikeDescriptor& descriptor, bool append = true) {
    std::ofstream file(path, append ? std::ios::app : std::ios::out);
    if (!file.is_open()) return false;

    file << descriptor.label;
    for (double val : descriptor.zernike)
        file << "," << val;
    file << "\n";

    return true;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_visio_1p3_MainActivity_guardarDescriptorZernikeDesdeApp(
        JNIEnv *env, jobject thiz, jobject bitmap, jstring jlabel, jstring jpath) {

    const char* label = env->GetStringUTFChars(jlabel, nullptr);
    const char* path = env->GetStringUTFChars(jpath, nullptr);

    AndroidBitmapInfo info;
    void* pixels;

    if (AndroidBitmap_getInfo(env, bitmap, &info) < 0 ||
        AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) {
        env->ReleaseStringUTFChars(jlabel, label);
        env->ReleaseStringUTFChars(jpath, path);
        return JNI_FALSE;
    }

    cv::Mat img(info.height, info.width, CV_8UC4, pixels);
    AndroidBitmap_unlockPixels(env, bitmap);

    // Procesar binarización y contornos
    cv::Mat gray, bin;
    cv::cvtColor(img, gray, cv::COLOR_RGBA2GRAY);
    cv::threshold(gray, bin, 170, 255, cv::THRESH_BINARY);

    std::vector<std::vector<cv::Point>> contornos;
    cv::findContours(bin, contornos, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contornos.empty()) {
        env->ReleaseStringUTFChars(jlabel, label);
        env->ReleaseStringUTFChars(jpath, path);
        return JNI_FALSE;
    }

    auto max_contorno = *std::max_element(contornos.begin(), contornos.end(),
                                          [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                                              return cv::contourArea(a) < cv::contourArea(b);
                                          });

    cv::Mat filled = cv::Mat::zeros(bin.size(), CV_8UC1);
    cv::drawContours(filled, std::vector<std::vector<cv::Point>>{max_contorno}, -1, cv::Scalar(255), cv::FILLED);

    std::vector<double> zernike = calculateZernikeMoments(filled);

    ZernikeDescriptor descriptor;
    descriptor.label = label;
    descriptor.zernike = zernike;

    bool exito = guardarDescriptorZernikeCSV(path, descriptor, true);

    env->ReleaseStringUTFChars(jlabel, label);
    env->ReleaseStringUTFChars(jpath, path);
    return exito ? JNI_TRUE : JNI_FALSE;
}