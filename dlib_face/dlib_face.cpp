// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

This example program shows how to find frontal human faces in an image and
estimate their pose.  The pose takes the form of 68 landmarks.  These are
points on the face such as the corners of the mouth, along the eyebrows, on
the eyes, and so forth.



This face detector is made using the classic Histogram of Oriented
Gradients (HOG) feature combined with a linear classifier, an image pyramid,
and sliding window detection scheme.  The pose estimator was created by
using dlib's implementation of the paper:
One Millisecond Face Alignment with an Ensemble of Regression Trees by
Vahid Kazemi and Josephine Sullivan, CVPR 2014
and was trained on the iBUG 300-W face landmark dataset.

Also, note that you can train your own models using dlib's machine learning
tools.  See train_shape_predictor_ex.cpp to see an example.




Finally, note that the face detector is fastest when compiled with at least
SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
chip then you should enable at least SSE2 instructions.  If you are using
cmake to compile this program you can enable them by using one of the
following commands when you create the build project:
cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
This will set the appropriate compiler options for GCC, clang, Visual
Studio, or the Intel compiler.  If you are using another compiler then you
need to consult your compiler's manual to determine how to enable these
instructions.  Note that AVX is the fastest but requires a CPU from at least
2011.  SSE4 is the next fastest and is supported by most current machines.
*/

#define DLIB_JPEG_SUPPORT


#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>

using namespace dlib;
using namespace std;

//#include <opencv2/highgui/highgui.hpp>
//#include "cstringt.h"
#include "io.h"
#include <string>
#include <vector>
#include <fstream>

//#include <opencv/cvaux.hpp>
//using namespace::cv;

std::vector<string>filenamesGlobal;
std::vector<string> getFilesName(string path){
	//文件句柄  
	long   hFile = 0;
	//文件信息  
	struct _finddata_t fileinfo;
	string p;
	std::vector<string>filenames;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//如果是目录,迭代之  
			//如果不是,加入列表  
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFilesName(p.assign(path).append("\\").append(fileinfo.name));
			}
			else
			{
				filenames.push_back(fileinfo.name);  // 直接获取名字即可
				//files.push_back(p.assign(path).append("\\").append(fileinfo.name));  
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
	return filenames;
}

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{
	char filePath[150] = "E:\\CASIA-COLOR";
	
		filenamesGlobal = getFilesName(filePath);
		argc = filenamesGlobal.size() + 1;
		argv[1] = "shape_predictor_68_face_landmarks.dat";
		
		//*argv = new char[1000];
		//for (int i = 2; i < argc; i++)
		//{
		//	cout << filenamesGlobal[i].c_str();
		//	//strcat(filePath, filenamesGlobal[i].c_str());
		//	strcpy(argv[i], filenamesGlobal[i].c_str());
		//	//argv[i] = filenamesGlobal[i].c_str();
		//}
	
	

	try
	{
		// This example takes in a shape model file and then a list of images to
		// process.  We will take these filenames in as command line arguments.
		// Dlib comes with example images in the examples/faces folder so give
		// those as arguments to this program.
		if (argc == 1)
		{
			cout << "Call this program like this:" << endl;
			cout << "./face_landmark_detection_ex shape_predictor_68_face_landmarks.dat faces/*.jpg" << endl;
			cout << "\nYou can get the shape_predictor_68_face_landmarks.dat file from:\n";
			cout << "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
			//return 0;
		}

		// We need a face detector.  We will use this to get bounding boxes for
		// each face in an image.
		frontal_face_detector detector = get_frontal_face_detector();
		// And we also need a shape_predictor.  This is the tool that will predict face
		// landmark positions given an image and face bounding box.  Here we are just
		// loading the model from the shape_predictor_68_face_landmarks.dat file you gave
		// as a command line argument.
		shape_predictor sp;
		deserialize(argv[1]) >> sp;


		//image_window win, win_faces;

		// Loop over all the images provided on the command line.
		for (int i = 0; i < filenamesGlobal.size(); ++i)
		{
			char filePath[150] = "E:\\CASIA-COLOR\\";
			strcat(filePath, filenamesGlobal[i].c_str());
			//cout << "processing image " << filenamesGlobal[i] << endl;
			array2d<rgb_pixel> img;
				//cout << filenamesGlobal[i].c_str();
				load_image(img, filePath);
			
			// Make the image larger so we can detect small faces.
			pyramid_up(img);

			// Now tell the face detector to give us a list of bounding boxes
			// around all the faces in the image.
			clock_t start2, end2;
			start2 = clock();
			std::vector<rectangle> dets = detector(img);
			end2 = (double)(1000 * (clock() - start2) / CLOCKS_PER_SEC);
			//cout << "face_detection_time:" << end2 <<"ms"<< std::endl;

			cout << "Number of faces detected: " << dets.size() << endl;

			// Now we will go ask the shape_predictor to tell us the pose of
			// each face we detected.
			std::vector<full_object_detection> shapes;
			for (unsigned long j = 0; j < dets.size(); ++j)
			{
				clock_t start2, end2;
				start2 = clock();
				std::vector<rectangle> dets = detector(img);

				full_object_detection shape = sp(img, dets[j]);

				end2 = (double)(1000 * (clock() - start2) / CLOCKS_PER_SEC);
				//cout << "face_aligement_time:" << end2 <<"ms"<< std::endl;

				//cout << "number of parts: " << shape.num_parts() << endl;
				//cout << "pixel position of first part:  " << shape.part(0) << endl;
				//cout << "pixel position of second part: " << shape.part(1) << endl;
				// You get the idea, you can get all the face part locations if
				// you want them.  Here we just store them in shapes so we can
				// put them on the screen.
				shapes.push_back(shape);
			}

			// Now let's view our face poses on the screen.
			//win.clear_overlay();
			//win.set_image(img);
			//win.add_overlay(render_face_detections(shapes));

			// We can also extract copies of each face that are cropped, rotated upright,
			// and scaled to a standard size as shown here:
			dlib::array<array2d<rgb_pixel> > face_chips;
			extract_image_chips(img, get_face_chip_details(shapes), face_chips);
			//win_faces.set_image(tile_images(face_chips));
			char savePath[150] = "C:\\Users\\Stefan\\Desktop\\dlib_face\\dlib_face\\result\\";
			strcat(savePath, filenamesGlobal[i].c_str());
			if (dets.size())
			{
				save_jpeg(tile_images(face_chips), savePath);
			}
			
			//cout << "Hit enter to process the next image..." << endl;
			//cin.get();
		}
		system("pause");
	}
	catch (exception& e)
	{
		cout << "\nexception thrown!" << endl;
		cout << e.what() << endl;
	}
}

// ----------------------------------------------------------------------------------------

