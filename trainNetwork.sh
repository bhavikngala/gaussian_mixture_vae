#!/usr/bin/env bash
if [ $1 == 'mnist' ]
then
	echo 'mnist'
	python train.py --K 10 --x-size 200 --w-size 150 --dataset $1 &
elif [ $# == 2 ] && [ $1 == 'tvsum' ] && [ $2 == 'gist' ]
then
	echo 'tvsum and gist'
	python train.py --K 10 --x-size 128 --w-size 128 --dataset $1 --feature-type $2 --epochs 100 --continuous &
elif [ $1 == 'tvsum' ]
then
	echo 'tvsum'
	python train.py --K 10 --x-size 256 --w-size 128 --dataset $1 --epochs 500 --continuous &
elif [ $1 == 'toy' ]
then
	echo 'toy'
	python train.py --K 3 --x-size 256 --w-size 128 --dataset $1 --epochs 500 &
elif [ $1 == 'spiral' ]
then
	echo 'spiral'
	python train.py --K 8 --x-size 2 --w-size 2 --dataset $1 --epochs 500 --continuous --batch-size 200 &
fi