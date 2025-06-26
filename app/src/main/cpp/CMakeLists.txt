cmake_minimum_required(VERSION 3.22.1)

project("visio_p3")

# Incluir cabeceras locales de OpenCV
include_directories(include)

# Importar OpenCV .so
add_library(opencv_java4 SHARED IMPORTED)
set_target_properties(opencv_java4 PROPERTIES
        IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libopencv_java4.so
)

# Crear tu biblioteca nativa
add_library(native-lib SHARED
        native-lib.cpp)

# Encontrar librerías del sistema
find_library(log-lib log)
find_library(android-lib android)  # <--- Esta línea está perfecta ahora

# Enlazar todas las dependencias
target_link_libraries(native-lib
        opencv_java4
        -ljnigraphics
        ${log-lib}
        ${android-lib})  # ✅ ENLAZADO CORRECTO
