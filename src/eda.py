from data import get_train_loader, get_test_loader

train_loader = get_train_loader()
test_loader = get_test_loader()

for batch in train_loader:
    print(len(batch))
    break

for batch in test_loader:
    print(len(batch))
    break