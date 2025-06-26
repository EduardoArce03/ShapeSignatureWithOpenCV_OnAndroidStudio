package com.example.visio_p3;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Bitmap;
import android.os.Bundle;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import com.example.visio_p3.databinding.ActivityMainBinding;

public class MainActivity extends AppCompatActivity {

    // Used to load the 'visio_p3' library on application startup.
    static {
        //System.loadLibrary("visio_p3");
        System.loadLibrary("opencv_java4");
        System.loadLibrary("native-lib");
    }

    public native String detectarFigura(Bitmap bitmap);

    private ActivityMainBinding binding;
    private DrawingView drawingView;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_main);
        drawingView = findViewById(R.id.drawingView);
        Button btnDetect = findViewById(R.id.btnDetect);

        btnDetect.setOnClickListener(v -> {
            Bitmap bmp = drawingView.getBitmap();
            String result = detectarFigura(bmp);
            Toast.makeText(this, result, Toast.LENGTH_LONG).show();
        });
    }

    /**
     * A native method that is implemented by the 'visio_p3' native library,
     * which is packaged with this application.
     */
    public native String stringFromJNI();
}