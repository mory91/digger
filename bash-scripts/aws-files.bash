#!/bin/bash
rsync -vzSP --files-from=rsync-files ec2-user@ec2-13-59-57-193.us-east-2.compute.amazonaws.com:ogomon/records ../data/$1/primary/
rsync -vzSP --files-from=rsync-files ec2-user@ec2-3-138-61-101.us-east-2.compute.amazonaws.com:ogomon/records ../data/$1/core/
rsync -vzSP --files-from=rsync-files ec2-user@ec2-3-136-27-201.us-east-2.compute.amazonaws.com:ogomon/records ../data/$1/task/
