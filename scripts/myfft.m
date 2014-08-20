function result = myfft(signal, Fs)
	dt = 1/Fs;   
	N = size(signal, 2);
	StopTime = N * dt;
	t = (0:dt:StopTime-dt);
	dF = Fs/N;                      % hertz
	f = 0:dF:Fs-dF;


	hann_window = transpose(hanning(N));
    %hann_window = transpose(hamming(N));
	avg = mean(signal);
	
	str = sprintf('Sampling Freq: %f Number of Samples: %d StopTime: %fi Avg: %f dF: %f', Fs, N, StopTime, avg, dF);
	
	disp(str);
	%signal_filtered = (signal - avg);% .* hann_window;
	signal_filtered = (signal - avg) .* hann_window;
    signal_filtered = filter(ones(1,2)/2, 1, signal_filtered);
	%signal_filtered = detrend(signal) .* hann_window;

	ft_signal = fft(signal_filtered);
	power_ft_signal = abs(ft_signal) / N;
	
    pxx= periodogram(signal);
	figure;
	plot(t, signal); title('original'); grid on;
	figure;
	plot(t, signal_filtered); title('filtered'); grid on;
	figure;
	plot(t, detrend(signal)); title('detrend'); grid on;
	figure;
	plot(f(1:N/2), power_ft_signal(1:N/2), 'b--+'); title('|ft|'); grid on;

    figure;
    plot(f(1:N/2), pxx(1:N/2)); title('periodogram'); grid on;

	result = power_ft_signal;
