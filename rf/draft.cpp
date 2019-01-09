//LARGE_INTEGER frequency;        // ticks per second
//LARGE_INTEGER t1, t2;           // ticks
//double elapsedTime;
//
//if (MEASURE) QueryPerformanceFrequency(&frequency);
//if (MEASURE) QueryPerformanceCounter(&t1);
//
//if (MEASURE) {
//	QueryPerformanceCounter(&t2);
//	elapsedTime = (t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart;
//	printf("Initialize pyramidal images: %.2f ms\n", elapsedTime);
//}