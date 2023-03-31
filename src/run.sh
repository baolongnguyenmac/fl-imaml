# fed avg mnist
python main.py --global_epochs=200 --local_epochs=1 --local_lr=0.001 --dataset='mnist' --model='mnist' --algorithm='fed_avg' --clients_per_round=5
python main.py --global_epochs=200 --local_epochs=5 --local_lr=0.001 --dataset='mnist' --model='mnist' --algorithm='fed_avg' --clients_per_round=5
python main.py --global_epochs=200 --local_epochs=10 --local_lr=0.001 --dataset='mnist' --model='mnist' --algorithm='fed_avg' --clients_per_round=5
python main.py --global_epochs=200 --local_epochs=15 --local_lr=0.001 --dataset='mnist' --model='mnist' --algorithm='fed_avg' --clients_per_round=5

# fed avg cifar
python main.py --global_epochs=200 --local_epochs=1  --local_lr=0.001 --dataset='cifar' --model='cifar' --algorithm='fed_avg' --clients_per_round=5
python main.py --global_epochs=200 --local_epochs=5  --local_lr=0.001 --dataset='cifar' --model='cifar' --algorithm='fed_avg' --clients_per_round=5
python main.py --global_epochs=200 --local_epochs=10 --local_lr=0.001 --dataset='cifar' --model='cifar' --algorithm='fed_avg' --clients_per_round=5
python main.py --global_epochs=200 --local_epochs=15 --local_lr=0.001 --dataset='cifar' --model='cifar' --algorithm='fed_avg' --clients_per_round=5


# fed maml vs imaml (1, 5, 10 CG step) mnist
python main.py --global_epochs=200 --global_lr=0.0001 --local_epochs=1  --local_lr=0.001 --dataset='mnist' --model='mnist' --algorithm='fed_maml'  --clients_per_round=5
python main.py --global_epochs=200 --global_lr=0.0001 --local_epochs=1  --local_lr=0.001 --dataset='mnist' --model='mnist' --algorithm='fed_imaml' --clients_per_round=5 --lambda_=2 --cg_step=1
python main.py --global_epochs=200 --global_lr=0.0001 --local_epochs=1  --local_lr=0.001 --dataset='mnist' --model='mnist' --algorithm='fed_imaml' --clients_per_round=5 --lambda_=2 --cg_step=5
python main.py --global_epochs=200 --global_lr=0.0001 --local_epochs=1  --local_lr=0.001 --dataset='mnist' --model='mnist' --algorithm='fed_imaml' --clients_per_round=5 --lambda_=2 --cg_step=10

python main.py --global_epochs=200 --global_lr=0.0001 --local_epochs=5  --local_lr=0.001 --dataset='mnist' --model='mnist' --algorithm='fed_maml'  --clients_per_round=5
python main.py --global_epochs=200 --global_lr=0.0001 --local_epochs=5  --local_lr=0.001 --dataset='mnist' --model='mnist' --algorithm='fed_imaml' --clients_per_round=5 --lambda_=2 --cg_step=1
python main.py --global_epochs=200 --global_lr=0.0001 --local_epochs=5  --local_lr=0.001 --dataset='mnist' --model='mnist' --algorithm='fed_imaml' --clients_per_round=5 --lambda_=2 --cg_step=5
python main.py --global_epochs=200 --global_lr=0.0001 --local_epochs=5  --local_lr=0.001 --dataset='mnist' --model='mnist' --algorithm='fed_imaml' --clients_per_round=5 --lambda_=2 --cg_step=10

python main.py --global_epochs=200 --global_lr=0.0001 --local_epochs=10 --local_lr=0.001 --dataset='mnist' --model='mnist' --algorithm='fed_maml'  --clients_per_round=5
python main.py --global_epochs=200 --global_lr=0.0001 --local_epochs=10 --local_lr=0.001 --dataset='mnist' --model='mnist' --algorithm='fed_imaml' --clients_per_round=5 --lambda_=2 --cg_step=1
python main.py --global_epochs=200 --global_lr=0.0001 --local_epochs=10 --local_lr=0.001 --dataset='mnist' --model='mnist' --algorithm='fed_imaml' --clients_per_round=5 --lambda_=2 --cg_step=5
python main.py --global_epochs=200 --global_lr=0.0001 --local_epochs=10 --local_lr=0.001 --dataset='mnist' --model='mnist' --algorithm='fed_imaml' --clients_per_round=5 --lambda_=2 --cg_step=10

python main.py --global_epochs=200 --global_lr=0.0001 --local_epochs=15 --local_lr=0.001 --dataset='mnist' --model='mnist' --algorithm='fed_maml'  --clients_per_round=5
python main.py --global_epochs=200 --global_lr=0.0001 --local_epochs=15 --local_lr=0.001 --dataset='mnist' --model='mnist' --algorithm='fed_imaml' --clients_per_round=5 --lambda_=2 --cg_step=1
python main.py --global_epochs=200 --global_lr=0.0001 --local_epochs=15 --local_lr=0.001 --dataset='mnist' --model='mnist' --algorithm='fed_imaml' --clients_per_round=5 --lambda_=2 --cg_step=5
python main.py --global_epochs=200 --global_lr=0.0001 --local_epochs=15 --local_lr=0.001 --dataset='mnist' --model='mnist' --algorithm='fed_imaml' --clients_per_round=5 --lambda_=2 --cg_step=10


# fed maml vs imaml (1, 5, 10 CG step) cifar
python main.py --global_epochs=200 --global_lr=0.0001 --local_epochs=1  --local_lr=0.001 --dataset='cifar'  --model='cifar' --algorithm='fed_maml'  --clients_per_round=5
python main.py --global_epochs=200 --global_lr=0.0001 --local_epochs=1  --local_lr=0.001 --dataset='cifar'  --model='cifar' --algorithm='fed_imaml' --clients_per_round=5 --lambda_=2 --cg_step=1
python main.py --global_epochs=200 --global_lr=0.0001 --local_epochs=1  --local_lr=0.001 --dataset='cifar'  --model='cifar' --algorithm='fed_imaml' --clients_per_round=5 --lambda_=2 --cg_step=5
python main.py --global_epochs=200 --global_lr=0.0001 --local_epochs=1  --local_lr=0.001 --dataset='cifar'  --model='cifar' --algorithm='fed_imaml' --clients_per_round=5 --lambda_=2 --cg_step=10

python main.py --global_epochs=200 --global_lr=0.0001 --local_epochs=5  --local_lr=0.001 --dataset='cifar'  --model='cifar' --algorithm='fed_maml'  --clients_per_round=5
python main.py --global_epochs=200 --global_lr=0.0001 --local_epochs=5  --local_lr=0.001 --dataset='cifar'  --model='cifar' --algorithm='fed_imaml' --clients_per_round=5 --lambda_=2 --cg_step=1
python main.py --global_epochs=200 --global_lr=0.0001 --local_epochs=5  --local_lr=0.001 --dataset='cifar'  --model='cifar' --algorithm='fed_imaml' --clients_per_round=5 --lambda_=2 --cg_step=5
python main.py --global_epochs=200 --global_lr=0.0001 --local_epochs=5  --local_lr=0.001 --dataset='cifar'  --model='cifar' --algorithm='fed_imaml' --clients_per_round=5 --lambda_=2 --cg_step=10

python main.py --global_epochs=200 --global_lr=0.0001 --local_epochs=10 --local_lr=0.001 --dataset='cifar'  --model='cifar' --algorithm='fed_maml'  --clients_per_round=5
python main.py --global_epochs=200 --global_lr=0.0001 --local_epochs=10 --local_lr=0.001 --dataset='cifar'  --model='cifar' --algorithm='fed_imaml' --clients_per_round=5 --lambda_=2 --cg_step=1
python main.py --global_epochs=200 --global_lr=0.0001 --local_epochs=10 --local_lr=0.001 --dataset='cifar'  --model='cifar' --algorithm='fed_imaml' --clients_per_round=5 --lambda_=2 --cg_step=5
python main.py --global_epochs=200 --global_lr=0.0001 --local_epochs=10 --local_lr=0.001 --dataset='cifar'  --model='cifar' --algorithm='fed_imaml' --clients_per_round=5 --lambda_=2 --cg_step=10

python main.py --global_epochs=200 --global_lr=0.0001 --local_epochs=15 --local_lr=0.001 --dataset='cifar'  --model='cifar' --algorithm='fed_maml'  --clients_per_round=5
python main.py --global_epochs=200 --global_lr=0.0001 --local_epochs=15 --local_lr=0.001 --dataset='cifar'  --model='cifar' --algorithm='fed_imaml' --clients_per_round=5 --lambda_=2 --cg_step=1
python main.py --global_epochs=200 --global_lr=0.0001 --local_epochs=15 --local_lr=0.001 --dataset='cifar'  --model='cifar' --algorithm='fed_imaml' --clients_per_round=5 --lambda_=2 --cg_step=5
python main.py --global_epochs=200 --global_lr=0.0001 --local_epochs=15 --local_lr=0.001 --dataset='cifar'  --model='cifar' --algorithm='fed_imaml' --clients_per_round=5 --lambda_=2 --cg_step=10
