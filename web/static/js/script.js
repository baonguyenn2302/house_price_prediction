const form = document.querySelector('#prediction-form');
const result = document.querySelector('#result');

document.querySelectorAll('.number-step').forEach((button) => {
  button.addEventListener('click', () => {
    const input = button.parentElement.querySelector('input[type="number"]');
    if (Number(button.dataset.step) > 0) input.stepUp();
    else input.stepDown();
    input.dispatchEvent(new Event('change', { bubbles: true }));
  });
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const submitButton = form.querySelector('button[type="submit"]');
  const payload = Object.fromEntries(new FormData(form).entries());
  submitButton.disabled = true;
  submitButton.textContent = 'Đang dự đoán...';
  try {
    const response = await fetch('/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Không thể dự đoán giá nhà.');
    result.className = 'result result-success';
    result.innerHTML = `<h2>📊 Kết quả dự đoán</h2><p>Giá nhà ước tính</p><p class="result-price"><strong>${data.formatted_prediction} ${data.unit}</strong></p><p class="result-note">Mô hình sử dụng: ${data.model_name}</p>`;
  } catch (error) {
    result.className = 'result result-error';
    result.textContent = `Lỗi từ Server API: ${error.message}`;
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = 'Dự đoán';
  }
});
