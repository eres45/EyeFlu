import React, { useState } from 'react';
import { Upload } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from './ui/alert';

const EyeFluDetection = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      if (file.size > 5 * 1024 * 1024) {
        setError('File size must be less than 5MB');
        return;
      }
      if (!['image/jpeg', 'image/png', 'image/gif'].includes(file.type)) {
        setError('Please upload a valid image file (JPEG, PNG, or GIF)');
        return;
      }
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setError(null);
    }
  };

  const handleSubmit = async () => {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      const response = await fetch('http://localhost:5000/api/detect', {
        method: 'POST',
        body: formData,
      });
      
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError('An error occurred while processing your image. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold mb-2">Eye Flu Detection Tool</h1>
        <p className="text-gray-600">Quick and accurate detection of eye flu symptoms</p>
      </div>

      <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
        <div className="mb-6">
          <div className="flex items-center justify-center w-full">
            <label className="flex flex-col items-center justify-center w-full h-64 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100">
              <div className="flex flex-col items-center justify-center pt-5 pb-6">
                <Upload className="w-10 h-10 mb-3 text-gray-400" />
                <p className="mb-2 text-sm text-gray-500">
                  <span className="font-semibold">Click to upload</span> or drag and drop
                </p>
                <p className="text-xs text-gray-500">JPEG, PNG, GIF (MAX. 5MB)</p>
              </div>
              <input 
                type="file" 
                className="hidden" 
                onChange={handleFileSelect}
                accept="image/jpeg,image/png,image/gif"
              />
            </label>
          </div>
        </div>

        {error && (
          <Alert variant="destructive" className="mb-6">
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {preview && (
          <div className="mb-6">
            <img 
              src={preview} 
              alt="Preview" 
              className="max-w-full h-auto mx-auto rounded-lg"
              style={{ maxHeight: '400px' }} 
            />
          </div>
        )}

        <button
          onClick={handleSubmit}
          disabled={loading || !selectedFile}
          className="w-full bg-blue-600 text-white p-3 rounded-lg font-semibold disabled:bg-gray-400 disabled:cursor-not-allowed hover:bg-blue-700 transition-colors"
        >
          {loading ? (
            <span className="flex items-center justify-center">
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2" />
              Processing...
            </span>
          ) : (
            'Analyze Image'
          )}
        </button>

        {result && (
          <div className="mt-6 p-4 bg-gray-50 rounded-lg">
            <h3 className="font-semibold text-lg mb-2">Results:</h3>
            <p className="text-gray-700">
              {result.hasEyeFlu 
                ? 'The image shows signs of eye flu. Please consult a healthcare professional.' 
                : 'No signs of eye flu detected.'}
            </p>
            <p className="mt-2 text-sm text-gray-500">
              Note: This is an automated analysis and should not be considered as a final diagnosis.
              Always consult with a healthcare professional for proper medical advice.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default EyeFluDetection;