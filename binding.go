package golight

/*
#cgo LDFLAGS: -l_lightgbm
#include <stdlib.h>
#include <stdint.h>
typedef void* BoosterHandle;
const char* LGBM_GetLastError();
int LGBM_BoosterCreateFromModelfile(const char* filename, int* out_num_iterations, BoosterHandle* out);
int LGBM_BoosterGetNumClasses(BoosterHandle handle, int* out_len);
int LGBM_BoosterFree(BoosterHandle handle);
int LGBM_BoosterCalcNumPredict(BoosterHandle handle, int num_row, int predict_type, int num_iteration, int64_t* out_len);
int LGBM_BoosterPredictForMat(BoosterHandle handle, const void* data, int data_type, int32_t nrow, int32_t ncol, int is_row_major, int predict_type, int num_iteration, const char* parameter, int64_t* out_len, double* out_result);
*/
import "C"
import (
	"unsafe"

	"github.com/pkg/errors"
)

const (
	C_API_PREDICT_NORMAL = 0
	C_API_DTYPE_FLOAT64  = 1
	C_API_IS_ROW_MAJOR   = 1
)

type Booster struct {
	handle C.BoosterHandle

	NumTotalIteration int
	NumClass          int
}

func safeCall(r C.int) error {
	if int(r) == -1 {
		msg := C.GoString(C.LGBM_GetLastError())
		return errors.New(msg)
	}
	return nil
}

func Load(filename string) (*Booster, error) {
	handle := C.BoosterHandle(nil)
	outNumIterations := C.int(0)
	cs := C.CString(filename)

	if err := safeCall(C.LGBM_BoosterCreateFromModelfile(cs, &outNumIterations, &handle)); err != nil {
		C.free(unsafe.Pointer(cs))
		return nil, errors.Wrap(err, "failed to create booster from model file")
	}
	C.free(unsafe.Pointer(cs))

	outNumClass := C.int(0)
	if err := safeCall(C.LGBM_BoosterGetNumClasses(handle, &outNumClass)); err != nil {
		return nil, errors.Wrap(err, "failed to get num classes")
	}

	return &Booster{
		handle:            handle,
		NumTotalIteration: int(outNumIterations),
		NumClass:          int(outNumClass),
	}, nil
}

func (b *Booster) Close() error {
	err := safeCall(C.LGBM_BoosterFree(b.handle))
	if err != nil {
		return errors.Wrap(err, "failed to free booster")
	}
	return nil
}

func (b *Booster) Predict(data []float64, numIteration int) ([]float64, error) {
	if numIteration > b.NumTotalIteration {
		numIteration = b.NumTotalIteration
	}
	nPreds := C.int64_t(0)
	if err := safeCall(C.LGBM_BoosterCalcNumPredict(b.handle, C.int(1), C.int(C_API_PREDICT_NORMAL), C.int(numIteration), &nPreds)); err != nil {
		return nil, errors.Wrap(err, "failed to calc num predict")
	}
	preds := make([]float64, nPreds)
	outNumPreds := C.int64_t(0)
	parameter := C.CString("")
	err := safeCall(C.LGBM_BoosterPredictForMat(
		b.handle,
		unsafe.Pointer((*C.double)(&data[0])),
		C.int(C_API_DTYPE_FLOAT64),
		C.int(1),
		C.int(len(data)),
		C.int(C_API_IS_ROW_MAJOR),
		C.int(C_API_PREDICT_NORMAL),
		C.int(numIteration),
		parameter,
		&outNumPreds,
		(*C.double)(&preds[0]),
	))
	C.free(unsafe.Pointer(parameter))
	if err != nil {
		return nil, errors.Wrap(err, "failed to predict for mat")
	}
	if int64(nPreds) != int64(outNumPreds) {
		return nil, errors.Errorf("wrong length for predict results. expected=%v, but got=%v", int64(nPreds), int64(outNumPreds))
	}
	return preds, nil
}
