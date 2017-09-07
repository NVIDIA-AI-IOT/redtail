// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Full license terms provided in LICENSE.md file.

#ifndef CAFFE_ROS_YOLO_PREDICTION_H
#define CAFFE_ROS_YOLO_PREDICTION_H

namespace caffe_ros
{
    struct ObjectPrediction
    {
        int   label;
        float prob;
        int   x;
        int   y;
        int   w;
        int   h;
    };

    std::vector<ObjectPrediction> getYoloPredictions(const float* predictions, size_t num_preds, int w_in, int h_in,
                                                     float prob_threshold = 0.1)
    {
        assert(predictions != nullptr);
        assert(num_preds > 0);
        assert(w_in > 0);
        assert(h_in > 0);
        assert(0 < prob_threshold && prob_threshold <= 1);

        const int grid_size     = 7;
        const int num_lab       = 20;
        const int num_box       = 2;
        const int num_box_coord = 4;

        assert(num_preds == grid_size * grid_size * (num_box * (num_box_coord  + 1) + num_lab));

        std::vector<ObjectPrediction> res;
        size_t icell = 0;
        for (int row = 0; row < grid_size; row++)
        {
            for (int col = 0; col < grid_size; col++, icell++)
            {
                // Find max conditional class probability for the current cell.
                auto cell_preds = predictions + icell * num_lab;
                auto it_max = std::max_element(cell_preds, cell_preds + num_lab);
                int  imax_p = it_max - cell_preds;
                assert(0 <= imax_p && imax_p < num_lab);
                float max_p = *(it_max);
                // Find a box with a max condidence prediction.
                auto  cell_box_scores = predictions + grid_size * grid_size * num_lab + icell * num_box;
                auto  it_box_max = std::max_element(cell_box_scores, cell_box_scores + num_box);
                int   imax_box   = it_box_max - cell_box_scores;
                assert(0 <= imax_box && imax_box < num_box);
                float box_score  = *(it_box_max);
                // Skip entries with conditional class probability below the threshold.
                if (box_score * max_p < prob_threshold)
                    continue;
                // Save box for the current cell.
                auto cell_box_coords = predictions + grid_size * grid_size * (num_lab + num_box) + (icell * num_box + imax_box) * num_box_coord;
                float x = (cell_box_coords[0] + col) / grid_size * w_in;
                float y = (cell_box_coords[1] + row) / grid_size * h_in;
                float w = std::max(cell_box_coords[2], 0.0f);
                float h = std::max(cell_box_coords[3], 0.0f);
                // Square the w/h as it was trained like that in YOLO.
                w *= w * w_in;
                h *= h * h_in;
                // x,y is the center of the box, find top left corner.
                x -= w / 2;
                y -= h / 2;
                // Make sure box coordinates are in the valid range.
                x  = std::min(std::max(x, 0.0f), (float)w_in - 1);
                y  = std::min(std::max(y, 0.0f), (float)h_in - 1);
                w  = std::min(w, w_in - x);
                h  = std::min(h, h_in - y);
                ObjectPrediction cur;
                cur.label = imax_p;
                cur.prob  = box_score * max_p;
                cur.x = (int)x;
                cur.y = (int)y;
                cur.w = (int)w;
                cur.h = (int)h;
                assert(0 <= cur.x && cur.x <  w_in);
                assert(0 <= cur.y && cur.y <  h_in);
                assert(0 <  cur.x + cur.w && cur.x + cur.w <= w_in);
                assert(0 <  cur.y + cur.h && cur.y + cur.h <= h_in);
                res.push_back(cur);
            }
        }
        assert(icell == grid_size * grid_size);
        return res;
    }

    // Filter objects according to IOU threshold.
    std::vector<ObjectPrediction> filterByIOU(std::vector<ObjectPrediction> src, float iou_threshold = 0.5)
    {
        assert(0 < iou_threshold && iou_threshold <= 1);
        
        // Filter out overlapping boxes.
        size_t i1 = 0;
        while (i1 < src.size())
        {
            auto   b1 = src[i1];
            size_t i2 = i1 + 1;
            while (i2 < src.size())
            {
                auto  b2 = src[i2];
                float b_union = b1.w * b1.h + b2.w * b2.h;
                assert(b_union > 0);
                float wi = std::max(std::min(b1.x + b1.w - b2.x, b2.x + b2.w - b1.x), 0);
                float hi = std::max(std::min(b1.y + b1.h - b2.y, b2.y + b2.h - b1.y), 0);
                float b_intersect = wi * hi;
                assert(0 <= b_intersect && b_intersect <= b_union);
                float iou = b_intersect / (b_union - b_intersect);
                // Remove box with IOU above threshold (e.g. "ambiguous" or "duplicate" box).
                if (iou > iou_threshold)
                    src.erase(src.begin() + i2);
                else
                    i2++;
            }
            i1++;
        }

        return src;
    }
}

#endif
