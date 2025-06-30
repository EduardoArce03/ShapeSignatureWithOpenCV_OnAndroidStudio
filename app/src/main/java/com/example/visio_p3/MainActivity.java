package com.example.visio_p3;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Bitmap;
import android.os.Bundle;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.example.visio_p3.databinding.ActivityMainBinding;

public class MainActivity extends AppCompatActivity {

    // Used to load the 'visio_p3' library on application startup.
    static {
        System.loadLibrary("opencv_java4");
        System.loadLibrary("native-lib");
    }

    private DrawingView drawingView;
    private ImageView imgResultado;
    private Button btnDetect;
    private TextView txtResultado;

    public native String detectarFigura(Bitmap bitmap, android.content.res.AssetManager assetManager);

    public native void procesarYMostrar(Bitmap input, Bitmap output);
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main); // ✅ NECESARIO antes de usar findViewById
        txtResultado = findViewById(R.id.txtResultado);

        drawingView = findViewById(R.id.drawingView);
        imgResultado = findViewById(R.id.imgResultado);
        btnDetect = findViewById(R.id.btnDetect);

        btnDetect.setOnClickListener(v -> {
            Bitmap entrada = drawingView.getBitmap();
            Bitmap salida = Bitmap.createBitmap(
                    entrada.getWidth(), entrada.getHeight(), Bitmap.Config.ARGB_8888);

            procesarYMostrar(entrada, salida);
            imgResultado.setImageBitmap(salida);

            // Mostrar resultado
            String resultado = detectarFigura(entrada, getAssets());
            txtResultado.setText(resultado); // <- O Toast.makeText(...) si prefieres
        });

    }

    /**
     * A native method that is implemented by the 'visio_p3' native library,
     * which is packaged with this application.
     */
    public native String stringFromJNI();
}