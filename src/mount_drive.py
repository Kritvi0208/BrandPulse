try:
    from google.colab import drive
    drive.mount('./gdrive/')
except Exception:
    print("Running in local environment; Google Drive mounting skipped.")