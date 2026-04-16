function reconstructed_image = apply_prednet(bias, kernel, decoded_features)
% Define the original dimensions of the feature map
    height = 64;
    width = 80;
    num_channels = 32;
    
    % Reshape the flattened decoded_features into [height, width, num_channels]
    reshaped_features = reshape(decoded_features, [height, width, num_channels]);
    
    % Initialize the output reconstructed image
    reconstructed_image = zeros(height, width, num_channels);
    
    % Loop over channels to apply convolution separately for each channel
    for ch = 1:num_channels
        % Convolve the decoded features with the kernel and add the bias
        kernel_ch = kernel(:, :, ch, :);  % Extract the kernel for this channel
        bias_ch = bias(ch);  % Extract the bias for this channel
        convolved_map = conv2(reshaped_features(:, :, ch), kernel_ch, 'same');  % Apply 2D convolution
        reconstructed_image(:, :, ch) = convolved_map + bias_ch;  % Add bias
    end
end
