using OpenCvSharp;
using Sdcb.OpenVINO.Natives;
using Sdcb.OpenVINO;
using System.Xml.Linq;
using System.Xml.XPath;
using System.Diagnostics;
using Sdcb.OpenVINO.Extensions.OpenCvSharp4;
using YoloObjectDetection;

public static class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("=== YOLOv8 OpenVINO Demo ===");
        Console.WriteLine("1. Classification Demo (static image)");
        Console.WriteLine("2. Object Detection with Webcam (real-time)");
        Console.Write("\nSelect option (1 or 2): ");

        string? choice = Console.ReadLine();

        if (choice == "2")
        {
            RunObjectDetectionWebcam();
        }
        else
        {
            RunClassificationDemo();
        }
    }

    static void RunObjectDetectionWebcam()
    {
        Console.WriteLine("\n=== Starting Object Detection with Webcam ===");

        string modelFile = @"./model/yolov8n.xml";
        string labelsFile = @"./coco-labels.txt";

        if (!File.Exists(modelFile))
        {
            Console.WriteLine($"\nERROR: Model file not found: {modelFile}");
            Console.WriteLine("\nPlease convert YOLOv8 detection model:");
            Console.WriteLine("1. Install ultralytics: pip install ultralytics");
            Console.WriteLine("2. Download YOLOv8n: wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt");
            Console.WriteLine("3. Convert to OpenVINO: yolo export model=yolov8n.pt format=openvino");
            Console.WriteLine("4. Copy yolov8n.xml and yolov8n.bin to ./model/ folder");
            return;
        }

        if (!File.Exists(labelsFile))
        {
            Console.WriteLine($"\nERROR: Labels file not found: {labelsFile}");
            return;
        }

        try
        {
            using var detector = new WebcamObjectDetection(
                modelPath: modelFile,
                labelsPath: labelsFile,
                confidenceThreshold: 0.25f,
                nmsThreshold: 0.45f
            );

            detector.Run(cameraIndex: 0);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"\nERROR: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
        }
    }

    static void RunClassificationDemo()
    {
        Console.WriteLine("\n=== Running Classification Demo ===");

        string modelFile = @"./model/yolov8n-cls.xml";
        string[] dicts = XDocument.Load(modelFile)
            .XPathSelectElement(@"/net/rt_info/model_info/labels")!.Attribute("value")!.Value
            .Split(' ');

        using Model rawModel = OVCore.Shared.ReadModel(modelFile);
        using PrePostProcessor pp = rawModel.CreatePrePostProcessor();
        using (PreProcessInputInfo inputInfo = pp.Inputs.Primary)
        {
            inputInfo.TensorInfo.Layout = Layout.NHWC;
            inputInfo.ModelInfo.Layout = Layout.NCHW;
            inputInfo.TensorInfo.ColorFormat = ov_color_format_e.BGR;
            inputInfo.Steps.ConvertColor(ov_color_format_e.RGB);
        }
        using Model m = pp.BuildModel();
        using CompiledModel cm = OVCore.Shared.CompileModel(m, "CPU");
        using InferRequest ir = cm.CreateInferRequest();

        Shape inputShape = m.Inputs.Primary.Shape;

        using Mat src = Cv2.ImRead(@"hen.jpg", ImreadModes.Color);
        Stopwatch stopwatch = new();
        using Mat resized = src.Resize(new Size(inputShape[2], inputShape[1]));
        using Mat f32 = new();
        resized.ConvertTo(f32, MatType.CV_32FC3, 1.0 / 255);

        using (Tensor input = f32.AsTensor())
        {
            ir.Inputs.Primary = input;
        }
        double preprocessTime = stopwatch.Elapsed.TotalMilliseconds;
        stopwatch.Restart();

        ir.Run();
        double inferTime = stopwatch.Elapsed.TotalMilliseconds;
        stopwatch.Restart();

        using (Tensor output = ir.Outputs.Primary)
        {
            ReadOnlySpan<float> data = output.GetData<float>();
            int maxIndex = MaxIndexOfSpan(data);
            double postProcessTime = stopwatch.Elapsed.TotalMilliseconds;
            stopwatch.Stop();

            Console.WriteLine($"class id={dicts[maxIndex]}, score={data[maxIndex]:F2}");
            double totalTime = preprocessTime + inferTime + postProcessTime;
            Console.WriteLine($"preprocess time: {preprocessTime:F2}ms");
            Console.WriteLine($"infer time: {inferTime:F2}ms");
            Console.WriteLine($"postprocess time: {postProcessTime:F2}ms");
            Console.WriteLine($"Total time: {totalTime:F2}ms");
        }
    }

    static int MaxIndexOfSpan(ReadOnlySpan<float> data)
    {
        // 参数校验
        if (data == null || data.Length == 0)
            throw new ArgumentException("The provided data span is null or empty.");

        // 初始化最大值及其索引
        int maxIndex = 0;
        float maxValue = data[0];

        // 遍历跨度查找最大值及其索引
        for (int i = 1; i < data.Length; i++)
        {
            if (data[i] > maxValue)
            {
                maxValue = data[i];
                maxIndex = i;
            }
        }

        // 返回最大值及其索引
        return maxIndex;
    }
}