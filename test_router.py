
from backend.snr_router import choose_path

print("Clean recording:", choose_path(18.5))
print("Noisy recording:", choose_path(4.2))
